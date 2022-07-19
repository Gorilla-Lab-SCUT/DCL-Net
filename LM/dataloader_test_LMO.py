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
import cv2

dict_id2name = {
    1: "ape",  
    5: "can",  
    6: "cat",
    8: "driller",  
    9: "duck",
    10: "eggbox",  
    11: "glue",  
    12: "holepuncher"  
}

def mask_to_bbox(mask, padding):
    mask = mask.astype(np.uint8)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    x = 0
    y = 0
    w = 0
    h = 0
    for contour in contours:
        tmp_x, tmp_y, tmp_w, tmp_h = cv2.boundingRect(contour)
        if tmp_w * tmp_h > w * h:
            x = tmp_x - int(padding/2)
            y = tmp_y - int(padding/2)
            w = tmp_w + padding
            h = tmp_h + padding
    return [x, y, w, h]

def get_linemod_to_occlusion_transformation(object_name):
        # https://github.com/ClayFlannigan/icp
        if object_name == 'ape':
            R = np.array([[0, -1,  0],
                         [0, 0, 1],
                         [-1, 0,  0]], dtype=np.float32)
            t = np.array([ 0.00464956, -0.04454319, -0.00454451], dtype=np.float32)
        elif object_name == 'can':
            R = np.array([[ 0, -1,  0],
                          [ 0, 0, 1],
                          [ -1,  0,  0]], dtype=np.float32)
            t = np.array([-0.009928,   -0.08974387, -0.00697199], dtype=np.float32)
        elif object_name == 'cat':
            R = np.array([[0, 1, 0],
                          [0, 0, 1],
                          [ 1,0, 0]], dtype=np.float32)
            t = np.array([-0.01460595, -0.05390565,  0.00600646], dtype=np.float32)
        elif object_name == 'driller':
            R = np.array([[0, -1,0],
                          [0, 0, 1],
                          [-1, 0,0]], dtype=np.float32)
            t = np.array([-0.00176942, -0.10016585,  0.00840302], dtype=np.float32)
        elif object_name == 'duck':
            R = np.array([[0, 1, 0],
                          [0, 0, 1],
                          [1, 0, 0]], dtype=np.float32)
            t = np.array([-0.00285449, -0.04044429,  0.00110274], dtype=np.float32)
        elif object_name == 'eggbox':
            R = np.array([[0, 1, 0],
                          [0, 0, 1],
                          [1, 0, 0]], dtype=np.float32)
            t = np.array([-0.01, -0.03, -0.00], dtype=np.float32)
        elif object_name == 'glue':
            R = np.array([[0, -1,  0],
                          [0, 0, 1],
                          [-1, 0,  0]], dtype=np.float32)
            t = np.array([-0.00144855, -0.07744411, -0.00468425], dtype=np.float32)
        elif object_name == 'holepuncher':
            R = np.array([[ 0, 1, 0],
                          [ 0, 0, 1],
                          [ 1, 0, 0]], dtype=np.float32)
            t = np.array([-0.00425799, -0.03734197,  0.00175619], dtype=np.float32)
        t = t.reshape((3, 1))
        return R, t

class Dataset():

    def __init__(self, mode, cfg, root='path_to_data'):
        self.npoint_inp = cfg.input_size
        self.npoint_tmp = cfg.tmp_size
        self.unit_voxel_extent = np.array(cfg.unit_voxel_extent).astype(np.float)
        self.voxel_num_limit = np.array(cfg.voxel_num_limit).astype(np.float)
        self.total_voxel_extent = self.voxel_num_limit * self.unit_voxel_extent
        self.voxelization_mode = cfg.voxelization_mode
        # use alignment_flipping to correct pose labels
        self.alignment_flipping = np.matrix([[1., 0., 0.],
                                             [0., -1., 0.],
                                             [0., 0., -1.]], dtype=np.float32)

        self.objlist = [1, 5, 6, 8, 9, 10, 11, 12] 
        self.symmetry_obj_idx = [5, 6]
        self.mode = mode

        self.list_rgb = []
        self.list_depth = []
        self.list_label = []
        self.list_rot   = []
        self.list_trans = []
        self.list_obj = []
        self.list_rank = []
        self.meta = {}
        self.root = root
        self.root_mask = "datasets/LMO_Masks"
        self.list_rgb_CAD = {}
        self.list_pc_CAD  = {}

        item_count = 0
        for item in self.objlist:
            path_cad = os.path.join('datasets/Linemod_preprocessed', "models", 'obj_{0}.ply'.format('%02d' % item))
            cad = o3d.io.read_triangle_mesh(path_cad)
            pcd = cad.sample_points_uniformly(number_of_points = self.npoint_tmp)
            self.list_rgb_CAD[item] = np.array(pcd.colors) - np.array([0.485, 0.456, 0.406])[np.newaxis,:]
            self.list_pc_CAD[item] = np.array(pcd.points)

            dir_valid_poses = os.path.join(root, "valid_poses", dict_id2name[item])
            poses = os.listdir(dir_valid_poses)
            for pose in poses:
                item_count += 1
                local_idx = int(pose.split(".")[0])
                path_pose = os.path.join(dir_valid_poses, pose)
                R, t, img_id = self.read_pose_and_img_id(path_pose, local_idx)
                R_lo, t_lo   = get_linemod_to_occlusion_transformation(dict_id2name[item])
                R = np.array(self.alignment_flipping * R, dtype=np.float32)
                t = np.array(self.alignment_flipping * t, dtype=np.float32)
                R = np.matmul(R, R_lo)
                
                self.list_rgb.append('{0}/RGB-D/rgb_noseg/{1}'.format(self.root, f"color_{str(img_id).zfill(5)}.png"))
                self.list_depth.append('{0}/RGB-D/depth_noseg/{1}'.format(self.root, f"depth_{str(img_id).zfill(5)}.png"))
                self.list_rot.append(R)
                self.list_trans.append(t.reshape(3))
                self.list_label.append('{0}/{1}/{2}.png'.format(self.root_mask, dict_id2name[item], local_idx))

                self.list_obj.append(item)

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
    def read_pose_and_img_id(self, filename, example_id):
        read_rotation = False
        read_translation = False
        R = []
        T = []
        with open(filename) as f:
            for line in f:
                if read_rotation:
                    R.append(line.split())
                    if len(R) == 3:
                        read_rotation = False
                elif read_translation:
                    T = line.split()
                    read_translation = False
                if line.startswith('rotation'):
                    read_rotation = True
                elif line.startswith('center'):
                    read_translation = True
        R = np.array(R, dtype=np.float32)
        T = np.array(T, dtype=np.float32).reshape((3, 1))
        img_id = int(line)
        return R, T, img_id
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        path_img = self.list_rgb[index]
        img = Image.open(self.list_rgb[index])
        depth = np.array(Image.open(self.list_depth[index]))
        label = np.array(Image.open(self.list_label[index]))
        obj = self.list_obj[index]
        rot = self.list_rot[index]
        trans= self.list_trans[index]

        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        if self.mode == 'eval':
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(1)))
        else:
            mask_label = ma.getmaskarray(ma.masked_equal(label, np.array([1, 1, 1])))[:, :, 0]
        mask = mask_label * mask_depth

        img = np.array(img)[:, :, :3]
        img_masked = img
        rmin, rmax, cmin, cmax = get_bbox(mask_to_bbox(mask_label, padding=0))
        target_r = np.resize(rot, (3, 3))
        target_t = np.array(trans)


        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) == 0:
            return torch.zeros(self.npoint_inp, 7), torch.zeros(self.npoint_inp, 3).long(), torch.zeros(self.npoint_tmp, 7), torch.zeros(self.npoint_tmp, 3).long(), torch.FloatTensor([-1]), torch.FloatTensor(target_r), torch.FloatTensor(target_t), torch.IntTensor([self.objlist.index(obj)]), path_img, torch.zeros(3)
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
        if np.sum(choose_idx)>0:
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

            centroid        = torch.FloatTensor(centroid)
            return feat_inp, voxel_index_inp.long(), feat_tmp, voxel_index_tmp.long(), torch.FloatTensor([symmetry_flag]), torch.FloatTensor(target_r), torch.FloatTensor(target_t), torch.IntTensor([self.objlist.index(obj)]), path_img, centroid

        else:

            return torch.zeros(self.npoint_inp, 7), torch.zeros(self.npoint_inp, 3).long(), torch.zeros(self.npoint_tmp, 7), torch.zeros(self.npoint_tmp, 3).long(), torch.FloatTensor([-1]), torch.zeros(3, 3), torch.zeros(3), torch.IntTensor([0]),path_img, torch.zeros(3)
    
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
            obj_idx                 = data[7].view(-1).int()
            path_img                = data[8][0]
            rot_gt                  = data[5]
            trans_gt                = data[6]
            return {
                "flags": sym_flags,
                "obj_idx" : obj_idx,
                "path_img": path_img,
                "rot_gt": rot_gt,
                "trans_gt": trans_gt
                
            }


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



