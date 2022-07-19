import open3d as o3d
import os
import torch
import numpy as np
from libs.pointgroup_ops.functions import pointgroup_ops

from PIL import Image
import numpy.ma as ma
import scipy.io as scio


class YCBDataset():

    def __init__(self, mode, cfg, root=None):
        self.npoint_inp = cfg.input_size
        self.npoint_tmp = cfg.tmp_size
        self.npoint_tmp_downsample = 1024
        self.unit_voxel_extent = np.array(cfg.unit_voxel_extent).astype(np.float)
        self.voxel_num_limit = np.array(cfg.voxel_num_limit).astype(np.float)
        self.total_voxel_extent = self.voxel_num_limit * self.unit_voxel_extent
        self.voxelization_mode = cfg.voxelization_mode
        self.mode = mode
        self.root = root
        self.path_mask = './datasets/YCBV_Masks/Masks_FFB6D'
        self.list = []
        self.path = './YCBV/utils_YCBV/test_data_list.txt'
        input_file = open(self.path)
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            self.list.append(input_line)
        input_file.close()

        self.length = len(self.list)
        print(self.length)

        class_file = open('./YCBV/utils_YCBV/classes.txt')
        class_id = 1
        self.list_rgb_CAD = {}
        self.list_pc_CAD  = {}
        self.list_pc_CAD_downsample = {}
        np.random.seed(1)
        while 1:
            class_input = class_file.readline()
            if not class_input:
                break
            
            path_cad  = os.path.join("./YCBV/CADs", class_input[:-1]+"_pc.ply")
            pcd = o3d.io.read_point_cloud(path_cad)
            if np.array(pcd.colors).shape[0] < self.npoint_tmp:
                choose_idx = np.random.choice(np.array(pcd.colors).shape[0], self.npoint_tmp)
            else:
                choose_idx = np.random.choice(np.array(pcd.colors).shape[0], self.npoint_tmp, replace=False)
            self.list_rgb_CAD[class_id] = np.array(pcd.colors)[choose_idx] - np.array([0.485, 0.456, 0.406])[np.newaxis,:]
            self.list_pc_CAD[class_id]  = np.array(pcd.points)[choose_idx] * 1000
            
            choose_idx_ds = np.random.choice(np.array(pcd.colors).shape[0], self.npoint_tmp_downsample, replace=False)
            self.list_pc_CAD_downsample[class_id]  = np.array(pcd.points)[choose_idx_ds] * 1000



            class_id += 1

        radius_all = 0.0
        count_num  = 0.0
        for item in self.list_pc_CAD:
            print(item, ": ", np.linalg.norm((self.list_pc_CAD[item]/ 1000.0), axis=1).max())
            radius_all += np.linalg.norm((self.list_pc_CAD[item]/ 1000.0), axis=1).max()
            count_num  += 1
        print("mean radius: ", radius_all / count_num)
        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        self.cam_cx = 312.9869
        self.cam_cy = 241.3109
        self.cam_fx = 1066.778
        self.cam_fy = 1067.487
        self.cam_scale = 10000.0

        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        self.minimum_num_pt = 50
        self.num_pt_mesh_small = 500
        self.num_pt_mesh_large = 2620
        self.symmetry_obj_idx = [12, 15, 18, 19, 20]


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = Image.open('{0}/{1}-color.png'.format(self.root, self.list[index]))
        depth = np.array(Image.open('{0}/{1}-depth.png'.format(self.root, self.list[index])))
        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        posecnn_meta = scio.loadmat('{0}/{1}.mat'.format(self.path_mask, '%06d' % index))
        label = np.array(posecnn_meta['labels'])
        posecnn_rois = np.array(posecnn_meta['rois'])

        gt_meta = scio.loadmat('{0}/{1}-meta.mat'.format(self.root, self.list[index]))
        gt_obj = gt_meta['cls_indexes'].flatten().astype(np.int32)

        ninstance = gt_obj.shape[0]
        all_flags = np.zeros((ninstance)).astype(np.int8)
        all_feats_inp = []
        all_voxel_indexes_inp = []
        all_feats_tmp = []
        all_voxel_indexes_tmp = []
        all_target_r  = []
        all_target_t  = []
        all_target_r_extra = []
        all_target_t_extra = []
        all_points_tmp_extra = []
        all_centroids        = []
        for idx in range(ninstance):
            if np.sum(posecnn_rois[:, 1]==gt_obj[idx]) == 0:
                all_flags[idx] = 0
                target_r = np.array(gt_meta['poses'][:, :, idx][:, 0:3])
                target_t = np.array([gt_meta['poses'][:, :, idx][:, 3:4].flatten()]).reshape((3))
                model_points = torch.FloatTensor(self.list_pc_CAD[int(gt_obj[idx])] / 1000.0)
                all_target_r_extra.append(torch.FloatTensor(target_r))
                all_target_t_extra.append(torch.FloatTensor(target_t))
                all_points_tmp_extra.append(model_points)
            else:
                all_flags[idx] = 1

                rmin, rmax, cmin, cmax = get_bbox(posecnn_rois, np.where(posecnn_rois[:, 1] == gt_obj[idx])[0][0])
                mask_label = ma.getmaskarray(ma.masked_equal(label, gt_obj[idx]))
                mask = mask_label * mask_depth

                target_r = np.array(gt_meta['poses'][:, :, idx][:, 0:3])
                target_t = np.array([gt_meta['poses'][:, :, idx][:, 3:4].flatten()]).reshape((3))

                choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
                
                if choose.shape[0] == 0:
                    all_flags[idx] = 0
                    target_r = np.array(gt_meta['poses'][:, :, idx][:, 0:3])
                    target_t = np.array([gt_meta['poses'][:, :, idx][:, 3:4].flatten()]).reshape((3))
                    model_points = torch.FloatTensor(self.list_pc_CAD[int(gt_obj[idx])] / 1000.0)
                    all_target_r_extra.append(torch.FloatTensor(target_r))
                    all_target_t_extra.append(torch.FloatTensor(target_t))
                    all_points_tmp_extra.append(model_points)
                    continue

                img_masked = np.array(img)[:, :, :3][rmin:rmax, cmin:cmax, :].astype(np.float32).reshape((-1, 3))
                img_masked = img_masked[choose, :]
                img_masked = img_masked/255.0 - np.array([0.485, 0.456, 0.406])[np.newaxis,:]

                depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)

                pt2 = depth_masked / self.cam_scale
                pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
                pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
                cloud = np.concatenate((pt0, pt1, pt2), axis=1)

                centroid = np.mean(cloud, axis=0)
                cloud = cloud - centroid[np.newaxis, :]
                target_t = target_t - centroid
                all_centroids.append(torch.tensor(centroid))

                choose_idx = (np.abs(cloud[:,0])<self.total_voxel_extent[0]*0.5) & (np.abs(cloud[:,1])<self.total_voxel_extent[1]*0.5) & (np.abs(cloud[:,2])<self.total_voxel_extent[2]*0.5)
                valid_num = np.sum(choose_idx)

                if valid_num>32:
                    cloud = cloud[choose_idx, :]
                    img_masked = img_masked[choose_idx, :]
                if cloud.shape[0]>self.npoint_inp:
                    choose_idx = np.random.choice(cloud.shape[0], self.npoint_inp, replace=False)
                else:
                    choose_idx = np.random.choice(cloud.shape[0], self.npoint_inp)
                cloud = torch.FloatTensor(cloud[choose_idx, :])
                img_masked = torch.FloatTensor(img_masked[choose_idx, :])

                feat_inp = torch.cat([torch.ones(self.npoint_inp, 1), img_masked, cloud], 1)
                voxel_index_inp = (cloud + self.total_voxel_extent[0]*0.5)/torch.FloatTensor(self.unit_voxel_extent)

                if valid_num<=32:
                    voxel_index_inp = torch.clamp(voxel_index_inp, min=0, max=self.voxel_num_limit[0]-1)

                
                model_points = torch.FloatTensor(self.list_pc_CAD[int(gt_obj[idx])] / 1000.0)
                model_colors = torch.FloatTensor(self.list_rgb_CAD[int(gt_obj[idx])])
                feat_tmp     = torch.cat([torch.ones(self.npoint_tmp, 1), model_colors, model_points], 1)
                voxel_index_tmp = (model_points + self.total_voxel_extent[0]*0.5)/torch.FloatTensor(self.unit_voxel_extent)

                all_feats_inp.append(feat_inp)
                all_voxel_indexes_inp.append(voxel_index_inp.long())
                all_feats_tmp.append(feat_tmp)
                all_voxel_indexes_tmp.append(voxel_index_tmp.long())
                all_target_r.append(torch.FloatTensor(target_r))
                all_target_t.append(torch.FloatTensor(target_t))
                
                all_target_r_extra.append(torch.FloatTensor(target_r))
                all_target_t_extra.append(torch.FloatTensor(target_t))
                all_points_tmp_extra.append(model_points)

        all_feats_inp = torch.stack(all_feats_inp)
        all_voxel_indexes_inp = torch.stack(all_voxel_indexes_inp)
        all_feats_tmp = torch.stack(all_feats_tmp)
        all_voxel_indexes_tmp = torch.stack(all_voxel_indexes_tmp)
        all_target_r  = torch.stack(all_target_r)
        all_target_t  = torch.stack(all_target_t)
        all_target_r_extra = torch.stack(all_target_r_extra)
        all_target_t_extra = torch.stack(all_target_t_extra)
        all_points_tmp_extra = torch.stack(all_points_tmp_extra)
        all_centroids        = torch.stack(all_centroids)

        b = all_feats_inp.size(0)

        batch_feats_inp = all_feats_inp.reshape(b*self.npoint_inp, 7)
        batch_voxel_indexes_inp = all_voxel_indexes_inp.reshape(b*self.npoint_inp, 3)
        batch_ids_inp = torch.arange(b).unsqueeze(1).repeat(1,self.npoint_inp).view(b*self.npoint_inp, 1).long()
        batch_voxel_indexes_inp = torch.cat([batch_ids_inp, batch_voxel_indexes_inp], 1)
        batch_occupied_voxels_inp, batch_p2v_maps_inp, batch_v2p_maps_inp = pointgroup_ops.voxelization_idx(batch_voxel_indexes_inp, b, self.voxelization_mode)

        batch_feats_tmp = all_feats_tmp.reshape(b*self.npoint_tmp, 7)
        batch_voxel_indexes_tmp = all_voxel_indexes_tmp.reshape(b*self.npoint_tmp, 3)
        batch_ids_tmp = torch.arange(b).unsqueeze(1).repeat(1,self.npoint_tmp).view(b*self.npoint_tmp, 1).long()
        batch_voxel_indexes_tmp = torch.cat([batch_ids_tmp, batch_voxel_indexes_tmp], 1)
        batch_occupied_voxels_tmp, batch_p2v_maps_tmp, batch_v2p_maps_tmp = pointgroup_ops.voxelization_idx(batch_voxel_indexes_tmp, b, self.voxelization_mode)

        batch_offsets = (torch.arange(b+1)*1024).int()
        voxel_num_limit = torch.tensor(self.voxel_num_limit)

        return {
                "inp":{
                    "feats"          : batch_feats_inp,
                    "occupied_voxels": batch_occupied_voxels_inp,
                    "p2v_maps"       : batch_p2v_maps_inp,
                    "v2p_maps"       : batch_v2p_maps_inp,
                },
                "tmp":{
                    "feats"          : batch_feats_tmp,
                    "occupied_voxels": batch_occupied_voxels_tmp,
                    "p2v_maps"       : batch_p2v_maps_tmp,
                    "v2p_maps"       : batch_v2p_maps_tmp,
                },
                "labels":{
                    "rot_gt"  : all_target_r,
                    "trans_gt": all_target_t,
                },
                "extra":{
                    "rot_gt_extra": all_target_r_extra,
                    "trans_gt_extra": all_target_t_extra,
                    "points_tmp_extra": all_points_tmp_extra
                },
                "all_centroids"  : all_centroids,
                "batch_offsets"  : batch_offsets,
                "voxel_num_limit": voxel_num_limit,
                "obj_idx"        : torch.IntTensor(gt_obj-1),
                "all_flags"      : torch.IntTensor(all_flags),
                "flags"          : torch.IntTensor([-1]),
                "path_img"       : '{0}/{1}-color.png'.format(self.root, self.list[index])

            }
    def collate(self, data_per_image):
        return data_per_image[0]


border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640
def get_bbox(posecnn_rois, idx):
    rmin = np.max([int(posecnn_rois[idx][3])+1, 0])
    rmax = np.min([int(posecnn_rois[idx][5])-1, img_width])
    cmin = np.max([int(posecnn_rois[idx][2])+1, 0])
    cmax = np.min([int(posecnn_rois[idx][4])-1, img_length])

    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax

