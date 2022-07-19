import os
import torch
from torch.utils.data.dataloader import default_collate
import numpy as np
import math
from libs.pointgroup_ops.functions import pointgroup_ops
from transforms3d.euler import euler2mat

from PIL import Image
import random
import numpy.ma as ma
import scipy.io as scio
import open3d as o3d


class Dataset():

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

        if mode == 'train':
            self.path = './YCBV/utils_YCBV/train_data_list.txt'
        elif mode == 'test':
            self.path = './YCBV/utils_YCBV/test_data_list.txt'

        self.list = []
        self.real = []
        self.syn = []
        input_file = open(self.path)
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            if input_line[:5] == 'data/':
                self.real.append(input_line)
            else:
                self.syn.append(input_line)
            self.list.append(input_line)
        input_file.close()

        self.length = len(self.list)

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
        self.radius_obj = {}
        for item in self.list_pc_CAD:
            print(item, ": ", np.linalg.norm((self.list_pc_CAD[item]/ 1000.0), axis=1).max())
            self.radius_obj[item] = np.linalg.norm((self.list_pc_CAD[item]/ 1000.0), axis=1).max()


        self.cam_cx_1 = 312.9869
        self.cam_cy_1 = 241.3109
        self.cam_fx_1 = 1066.778
        self.cam_fy_1 = 1067.487

        self.cam_cx_2 = 323.7872
        self.cam_cy_2 = 279.6921
        self.cam_fx_2 = 1077.836
        self.cam_fy_2 = 1078.189

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        self.minimum_num_pt = 50
        self.symmetry_obj_idx = [12, 15, 18, 19, 20]

        print(len(self.list))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        path_img = self.list[index]
        img = Image.open('{0}/{1}-color.png'.format(self.root, self.list[index]))
        depth = np.array(Image.open('{0}/{1}-depth.png'.format(self.root, self.list[index])))
        label = np.array(Image.open('{0}/{1}-label.png'.format(self.root, self.list[index])))
        meta = scio.loadmat('{0}/{1}-meta.mat'.format(self.root, self.list[index]))
        obj = meta['cls_indexes'].flatten().astype(np.int32)        

        if self.list[index][:8] != 'data_syn' and int(self.list[index][5:9]) >= 60:
            cam_cx = self.cam_cx_2
            cam_cy = self.cam_cy_2
            cam_fx = self.cam_fx_2
            cam_fy = self.cam_fy_2
        else:
            cam_cx = self.cam_cx_1
            cam_cy = self.cam_cy_1
            cam_fx = self.cam_fx_1
            cam_fy = self.cam_fy_1

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))

        while 1:
            idx = np.random.randint(0, len(obj))
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, obj[idx]))
            mask = mask_label * mask_depth
            if len(mask.nonzero()[0]) > self.minimum_num_pt:
                break

        rmin, rmax, cmin, cmax = get_bbox(mask_label)
        target_r = meta['poses'][:, :, idx][:, 0:3]
        target_t = np.array([meta['poses'][:, :, idx][:, 3:4].flatten()]).reshape((3))

        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) < self.minimum_num_pt:
            return torch.zeros(self.npoint_inp, 7), torch.zeros(self.npoint_inp, 3).long(), torch.zeros(self.npoint_tmp, 7), torch.zeros(self.npoint_tmp, 3).long(), torch.FloatTensor([-1]), torch.zeros(3, 3), torch.zeros(3), torch.IntTensor([-1]), path_img

        img_masked = np.array(img)[:, :, :3][rmin:rmax, cmin:cmax, :].astype(np.float32).reshape((-1, 3))
        img_masked = img_masked[choose, :]
        img_masked = img_masked/255.0 - np.array([0.485, 0.456, 0.406])[np.newaxis,:]

        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)

        cam_scale = meta['factor_depth'][0][0]
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)


        centroid = np.mean(cloud, axis=0)
        cloud = cloud - centroid[np.newaxis, :]
        target_t = target_t - centroid

        if self.mode == 'train':
            a1 = np.random.uniform(-math.pi/36.0, math.pi/36.0)
            a2 = np.random.uniform(-math.pi/36.0, math.pi/36.0)
            a3 = np.random.uniform(-math.pi/36.0, math.pi/36.0)
            aug_r = euler2mat(a1, a2, a3)
            aug_r = torch.FloatTensor(aug_r)
            cloud = torch.FloatTensor(cloud)
            target_t = torch.FloatTensor(target_t)
            target_r = torch.FloatTensor(target_r)

            cloud = (cloud - target_t[np.newaxis, :]) @ target_r
            target_t = target_t + torch.FloatTensor(np.array([random.uniform(-0.03, 0.03) for i in range(3)]))
            target_r = target_r @ aug_r
            cloud = cloud @ target_r.T + target_t[np.newaxis, :]
            cloud = cloud.numpy()
            target_t = target_t.numpy()
            target_r = target_r.numpy()

        model_points = torch.FloatTensor(self.list_pc_CAD[obj[idx]] / 1000.0)
        model_colors = torch.FloatTensor(self.list_rgb_CAD[obj[idx]])
        radius          = self.radius_obj[obj[idx]]


        if int(obj[idx]) - 1 in self.symmetry_obj_idx:
            symmetry_flag = 1
        else:
            symmetry_flag = 0

        choose_idx = (np.abs(cloud[:,0])<self.total_voxel_extent[0]*0.5) & (np.abs(cloud[:,1])<self.total_voxel_extent[1]*0.5) & (np.abs(cloud[:,2])<self.total_voxel_extent[2]*0.5)

        if np.sum(choose_idx)>self.minimum_num_pt:
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
            feat_tmp        = torch.cat([torch.ones(self.npoint_tmp, 1), model_colors, model_points], 1)
            voxel_index_tmp = (model_points + self.total_voxel_extent[0]*0.5)/torch.FloatTensor(self.unit_voxel_extent)
            return feat_inp, voxel_index_inp.long(), feat_tmp, voxel_index_tmp.long(), torch.FloatTensor([symmetry_flag]), torch.FloatTensor(target_r), torch.FloatTensor(target_t), torch.IntTensor([obj[idx]-1]), path_img, torch.FloatTensor([radius])

        else:

            return torch.zeros(self.npoint_inp, 7), torch.zeros(self.npoint_inp, 3).long(), torch.zeros(self.npoint_tmp, 7), torch.zeros(self.npoint_tmp, 3).long(), torch.FloatTensor([-1]), torch.zeros(3, 3), torch.zeros(3), torch.IntTensor([-1]), path_img, torch.FloatTensor([-1])

    def collate(self, indexes):
        data = default_collate(indexes)
        sym_flags = data[4][:, 0]

        flags = ~(sym_flags == -1)
        b = torch.sum(flags)
        if self.mode == "train" or (self.mode == "eval" and b>0):
            batch_feats_inp         = data[0][flags].reshape(b*self.npoint_inp, 7)
            batch_voxel_indexes_inp = data[1][flags].reshape(b*self.npoint_inp, 3)
            batch_feats_tmp         = data[2][flags].reshape(b*self.npoint_tmp, 7)
            batch_voxel_indexes_tmp = data[3][flags].reshape(b*self.npoint_tmp, 3)
            rot_gt                  = data[5][flags]
            trans_gt                = data[6][flags]
            sym_flags               = sym_flags[flags].float()
            obj_idx                 = data[7][flags]
            paths_img               = data[8]
            radius                  = data[9][flags].float()
            # process data
            batch_ids_inp = torch.arange(b).unsqueeze(1).repeat(1,self.npoint_inp).view(b*self.npoint_inp, 1).long()
            batch_voxel_indexes_inp = torch.cat([batch_ids_inp, batch_voxel_indexes_inp], 1)
            batch_occupied_voxels_inp, batch_p2v_maps_inp, batch_v2p_maps_inp = pointgroup_ops.voxelization_idx(batch_voxel_indexes_inp, b, self.voxelization_mode)

            batch_ids_tmp = torch.arange(b).unsqueeze(1).repeat(1,self.npoint_tmp).view(b*self.npoint_tmp, 1).long()
            batch_voxel_indexes_tmp = torch.cat([batch_ids_tmp, batch_voxel_indexes_tmp], 1)
            batch_occupied_voxels_tmp, batch_p2v_maps_tmp, batch_v2p_maps_tmp = pointgroup_ops.voxelization_idx(batch_voxel_indexes_tmp, b, self.voxelization_mode)

            batch_offsets = (torch.arange(b+1)*self.npoint_inp).int()
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
                    "rot_gt"  : rot_gt,
                    "trans_gt": trans_gt,
                    "obj_idx" : obj_idx,
                    "paths_img": paths_img,
                },
                
                "batch_offsets"  : batch_offsets,
                "voxel_num_limit": voxel_num_limit,
                "flags"          : sym_flags,
                "radius"         : radius

            }

        else:
            return {
                "flags": sym_flags,
            }




border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640

def get_bbox(label):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
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




