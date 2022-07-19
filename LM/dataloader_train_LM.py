import open3d as o3d
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
import yaml
import cv2

def mask_to_bbox(mask):
    mask = mask.astype(np.uint8)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    x = 0
    y = 0
    w = 0
    h = 0
    for contour in contours:
        tmp_x, tmp_y, tmp_w, tmp_h = cv2.boundingRect(contour)
        if tmp_w * tmp_h > w * h:
            x = tmp_x
            y = tmp_y
            w = tmp_w
            h = tmp_h
    return [x, y, w, h]
class Dataset():

    def __init__(self, mode, cfg, root="path_to_data"):
        self.npoint_inp = cfg.input_size
        self.npoint_tmp = cfg.tmp_size
        self.unit_voxel_extent = np.array(cfg.unit_voxel_extent).astype(np.float)
        self.voxel_num_limit = np.array(cfg.voxel_num_limit).astype(np.float)
        self.total_voxel_extent = self.voxel_num_limit * self.unit_voxel_extent
        self.voxelization_mode = cfg.voxelization_mode

        self.objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
        self.mode = mode

        self.list_rgb = []
        self.list_depth = []
        self.list_label = []
        self.list_obj = []
        self.list_rank = []
        self.meta = {}
        self.root = root
        self.list_rgb_CAD = {}
        self.list_pc_CAD  = {}
        
        self.dict_index_objs = {}

        item_count = 0
        index_count = 0
        for item in self.objlist:
            self.dict_index_objs[item] = []
            self.dict_index_objs[item].append(index_count)
            path_cad = os.path.join(self.root, "models", 'obj_{0}.ply'.format('%02d' % item))
            cad = o3d.io.read_triangle_mesh(path_cad)
            pcd = cad.sample_points_uniformly(number_of_points = self.npoint_tmp)
            self.list_rgb_CAD[item] = np.array(pcd.colors) - np.array([0.485, 0.456, 0.406])[np.newaxis,:]
            self.list_pc_CAD[item] = np.array(pcd.points)

            if self.mode == 'train':
                input_file = open('{0}/data/{1}/train.txt'.format(self.root, '%02d' % item))
            else:
                input_file = open('{0}/data/{1}/test.txt'.format(self.root, '%02d' % item))
            while 1:
                item_count += 1
                input_line = input_file.readline()
                if self.mode == 'test' and item_count % 10 != 0:
                    continue
                if not input_line:
                    break
                if input_line[-1:] == '\n':
                    input_line = input_line[:-1]
                self.list_rgb.append('{0}/data/{1}/rgb/{2}.png'.format(self.root, '%02d' % item, input_line))
                self.list_depth.append('{0}/data/{1}/depth/{2}.png'.format(self.root, '%02d' % item, input_line))
                if self.mode == 'eval':
                    self.list_label.append('{0}/segnet_results/{1}_label/{2}_label.png'.format(self.root, '%02d' % item, input_line))
                else:
                    self.list_label.append('{0}/data/{1}/mask/{2}.png'.format(self.root, '%02d' % item, input_line))

                self.list_obj.append(item)
                self.list_rank.append(int(input_line))
                index_count += 1
            self.dict_index_objs[item].append(index_count)
            meta_file = open('{0}/data/{1}/gt.yml'.format(self.root, '%02d' % item), 'r')
            self.meta[item] = yaml.load(meta_file)

            # print(np.max(self.list_pc_CAD[item]/ 1000.0, 0))
            # print(np.min(self.list_pc_CAD[item]/ 1000.0, 0))


            print("Object {0} buffer loaded".format(item))
            print(len(self.list_rgb))
        radius_all = 0.0
        count_num  = 0.0
        for item in self.list_pc_CAD:
            print(item, ": ", np.linalg.norm((self.list_pc_CAD[item]/ 1000.0), axis=1).max())
            radius_all += np.linalg.norm((self.list_pc_CAD[item]/ 1000.0), axis=1).max()
            count_num  += 1
        print("mean radius: ", radius_all / count_num)
        self.length = len(self.list_rgb)

        self.cam_cx = 325.26110
        self.cam_cy = 242.04899
        self.cam_fx = 572.41140
        self.cam_fy = 573.57043

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

        self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
        self.symmetry_obj_idx = [7, 8]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        path_img = self.list_rgb[index]
        img = Image.open(self.list_rgb[index])
        img = np.array(img)[:, :, :3]
        depth = np.array(Image.open(self.list_depth[index]))
        label = np.array(Image.open(self.list_label[index]))
        obj = self.list_obj[index]
        rank = self.list_rank[index]
        img, depth, label = self.occlude_with_another_object(img, depth, label, obj)

        if obj == 2:
            for i in range(0, len(self.meta[obj][rank])):
                if self.meta[obj][rank][i]['obj_id'] == 2:
                    meta = self.meta[obj][rank][i]
                    break
        else:
            meta = self.meta[obj][rank][0]

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        if self.mode == 'eval':
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))
        else:
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array([255, 255, 255])))[:, :, 0]
        mask = mask_label * mask_depth

        img_masked = img
        rmin, rmax, cmin, cmax = get_bbox(meta['obj_bb'])

        target_r = np.resize(np.array(meta['cam_R_m2c']), (3, 3))
        target_t = np.array(meta['cam_t_m2c']) / 1000.0


        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) == 0:
            return torch.zeros(self.npoint_inp, 7), torch.zeros(self.npoint_inp, 3).long(), torch.zeros(self.npoint_tmp, 7), torch.zeros(self.npoint_tmp, 3).long(), torch.FloatTensor([-1]), torch.zeros(3, 3), torch.zeros(3), torch.IntTensor([0]),"path_to_image", torch.zeros(3)
        img_masked = img_masked[rmin:rmax, cmin:cmax, :].astype(np.float32).reshape((-1, 3))
        img_masked = img_masked[choose, :]
        img_masked = img_masked/255.0 - np.array([0.485, 0.456, 0.406])[np.newaxis,:]

        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)

        cam_scale = 1.0
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
        pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)
        cloud = cloud / 1000.0

        centroid = np.mean(cloud, axis=0)
        cloud = cloud - centroid[np.newaxis, :]
        target_t = target_t - centroid

        if self.mode == 'train':
            a1 = np.random.uniform(-math.pi/36.0, math.pi/36.0)
            a2 = np.random.uniform(-math.pi/36.0, math.pi/36.0)
            a3 = np.random.uniform(-math.pi/36.0, math.pi/36.0)
            aug_r = euler2mat(a1, a2, a3)

            cloud = (cloud - target_t[np.newaxis, :]) @ target_r
            target_t = target_t + np.array([random.uniform(-0.03, 0.03) for i in range(3)])
            target_r = target_r @ aug_r
            cloud = cloud @ target_r.T + target_t[np.newaxis, :]


        model_points = torch.FloatTensor(self.list_pc_CAD[obj] / 1000.0)
        model_colors = torch.FloatTensor(self.list_rgb_CAD[obj])


        if self.objlist.index(obj) in self.symmetry_obj_idx:
            symmetry_flag = 1
        else:
            symmetry_flag = 0

        choose_idx = (np.abs(cloud[:,0])<self.total_voxel_extent[0]*0.5) & (np.abs(cloud[:,1])<self.total_voxel_extent[1]*0.5) & (np.abs(cloud[:,2])<self.total_voxel_extent[2]*0.5)
        if np.sum(choose_idx)>128 or self.mode=="eval":
            cloud = cloud[choose_idx, :]
            img_masked = img_masked[choose_idx, :]

            if cloud.shape[0]>self.npoint_inp:
                choose_idx = np.random.choice(cloud.shape[0], self.npoint_inp, replace=False)
            else:
                choose_idx = np.random.choice(cloud.shape[0], self.npoint_inp)
            cloud = torch.FloatTensor(cloud[choose_idx, :])
            img_masked = torch.FloatTensor(img_masked[choose_idx, :])

            feat_inp        = torch.cat([torch.ones(self.npoint_inp, 1), img_masked, cloud], 1)
            voxel_index_inp = (cloud + self.total_voxel_extent[0]*0.5)/torch.FloatTensor(self.unit_voxel_extent)

            feat_tmp        = torch.cat([torch.ones(self.npoint_tmp, 1), model_colors, model_points], 1)
            voxel_index_tmp = (model_points + self.total_voxel_extent[0]*0.5)/torch.FloatTensor(self.unit_voxel_extent)

            return feat_inp, voxel_index_inp.long(), feat_tmp, voxel_index_tmp.long(), torch.FloatTensor([symmetry_flag]), torch.FloatTensor(target_r), torch.FloatTensor(target_t), torch.IntTensor([self.objlist.index(obj)]), path_img, torch.FloatTensor(centroid)

        else:

            return torch.zeros(self.npoint_inp, 7), torch.zeros(self.npoint_inp, 3).long(), torch.zeros(self.npoint_tmp, 7), torch.zeros(self.npoint_tmp, 3).long(), torch.FloatTensor([-1]), torch.zeros(3, 3), torch.zeros(3), torch.IntTensor([0]),"path_to_image", torch.zeros(3)
    
    def collate(self, indexes):
        data = default_collate(indexes)
        sym_flags = data[4][:, 0]

        flags = ~(sym_flags == -1)
        b = torch.sum(flags)
        if self.mode == "train" or (self.mode == "eval" and b>0):
            
            obj_idx                 = data[7][flags].view(b).int()
            batch_feats_inp         = data[0][flags].reshape(b*self.npoint_inp, 7)
            batch_voxel_indexes_inp = data[1][flags].reshape(b*self.npoint_inp, 3)
            batch_feats_tmp         = data[2][flags].reshape(b*self.npoint_tmp, 7)
            batch_voxel_indexes_tmp = data[3][flags].reshape(b*self.npoint_tmp, 3)
            rot_gt                  = data[5][flags]
            trans_gt                = data[6][flags]
            sym_flags               = sym_flags[flags].float()
            path_img                = data[8][0]
            centriods               = data[9][flags]
            
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
                },
                
                "batch_offsets"  : batch_offsets,
                "voxel_num_limit": voxel_num_limit,
                "flags"          : sym_flags,
                "obj_idx"        : obj_idx,
                "path_img"       : [path_img],
                "centriods"      : centriods

            }

        else:
            return {
                "flags": sym_flags,
            }

    def get_other_idx(self, obj_idx):
        start = self.dict_index_objs[obj_idx][0]
        stop  = self.dict_index_objs[obj_idx][1]
        length_all = self.dict_index_objs[15][1]
        other_idx = random.choice( list(range(start)) + list(range(stop, length_all)) )
        return other_idx
    
    def occlude_with_another_object(self, image, depth, mask, obj_id):
        orig_image, orig_mask, orig_depth = image.copy(), mask.copy(), depth.copy()
        try:
            other_idx   = self.get_other_idx(obj_id)
            other_image = np.array(Image.open(self.list_rgb[other_idx]))
            other_depth = np.array(Image.open(self.list_depth[other_idx]))
            other_mask  = np.array(Image.open(self.list_label[other_idx]))
            
            other_ys, other_xs = np.nonzero(other_mask[:,:,0])
            other_ymin, other_ymax = np.min(other_ys), np.max(other_ys)
            other_xmin, other_xmax = np.min(other_xs), np.max(other_xs)
            ys, xs = np.nonzero(mask[:,:,0])
            ymin, ymax = np.min(ys), np.max(ys)
            xmin, xmax = np.min(xs), np.max(xs)
            other_mask  = other_mask[other_ymin:other_ymax+1, other_xmin:other_xmax+1]
            other_image = other_image[other_ymin:other_ymax+1, other_xmin:other_xmax+1]
            other_depth = other_depth[other_ymin:other_ymax+1, other_xmin:other_xmax+1]

            start_y = np.random.randint(ymin - other_mask.shape[0] + 1, ymax + 1)
            end_y = start_y + other_mask.shape[0]
            start_x = np.random.randint(xmin - other_mask.shape[1] + 1, xmax + 1)
            end_x = start_x + other_mask.shape[1]
            if start_y < 0:
                other_mask = other_mask[-start_y:]
                other_image = other_image[-start_y:]
                other_depth = other_depth[-start_y:]
                start_y = 0
            if end_y > image.shape[0]:
                end_y = image.shape[0]
                other_mask = other_mask[:end_y-start_y]
                other_image = other_image[:end_y-start_y]
                other_depth = other_depth[:end_y-start_y]
            if start_x < 0:
                other_mask = other_mask[-start_x:]
                other_image = other_image[-start_x:]
                other_depth = other_depth[-start_x:]
                start_x = 0
            if end_x > image.shape[0]:
                end_x = image.shape[0]
                other_mask = other_mask[:end_x-start_x]
                other_image = other_image[:end_x-start_x]
                other_depth = other_depth[:end_x-start_x]
            other_outline = (other_mask == 0)
            image[start_y:end_y, start_x:end_x] *= other_outline
            depth[start_y:end_y, start_x:end_x] *= other_outline[:,:,0]
            other_image[other_mask == 0] = 0
            other_depth[(other_mask == 0)[:,:,0]] = 0
            image[start_y:end_y, start_x:end_x] += other_image
            depth[start_y:end_y, start_x:end_x] += other_depth
            mask [start_y:end_y, start_x:end_x] *= (other_mask == 0)
            if mask.sum() >= 20:
                return image, depth, mask
            else:
                return orig_image, orig_depth, orig_mask
        except:
            return orig_image, orig_depth, orig_mask
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640

def get_bbox(bbox):
    bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
    if bbx[0] < 0:
        bbx[0] = 0
    if bbx[1] >= 480:
        bbx[1] = 479
    if bbx[2] < 0:
        bbx[2] = 0
    if bbx[3] >= 640:
        bbx[3] = 639
    rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
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
    if rmax > 480:
        delt = rmax - 480
        rmax = 480
        rmin -= delt
    if cmax > 640:
        delt = cmax - 640
        cmax = 640
        cmin -= delt
    return rmin, rmax, cmin, cmax


def ply_vtx(path):
    f = open(path)
    assert f.readline().strip() == "ply"
    f.readline()
    f.readline()
    N = int(f.readline().split()[-1])
    while f.readline().strip() != "end_header":
        continue
    pts = []
    for _ in range(N):
        pts.append(np.float32(f.readline().split()[:3]))
    return np.array(pts)



