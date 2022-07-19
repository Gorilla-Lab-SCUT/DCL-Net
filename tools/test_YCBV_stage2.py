import open3d as o3d
import os
import sys
import torch
import random
import logging
import gorilla
import argparse
import importlib
import numpy as np
import os.path as osp
import utils.tools_train as tools_train
from tqdm import tqdm



def get_parser():
    parser = argparse.ArgumentParser(
        description="YCBV test")

    parser.add_argument("--epoch",
                        type=int,
                        default=400,
                        help="Checkpoint of stage2 to be tested.")
    parser.add_argument("--epoch_stage1",
                        type=int,
                        default=400,
                        help="Checkpoint of stage1 to be tested.")

    parser.add_argument("--gpus",
                        type=str,
                        default="0",
                        help="GPUS to be used.")
    parser.add_argument("--model",
                        type=str,
                        default="DCL_Net",
                        help="The name of the model to be used in stage1.")
    parser.add_argument("--config",
                        type=str,
                        default="configs/config_YCBV.yaml",
                        help="Path to the config file used when training in stage2.")
    parser.add_argument("--config_stage1",
                        type=str,
                        default="configs/config_YCBV.yaml",
                        help="Path to the config file used when training in stage1.")
    parser.add_argument("--exp_id",
                        type=int,
                        default=10000,
                        help="The id of the experiment of stage2.")
    parser.add_argument("--exp_id_stage1",
                        type=int,
                        default=10000,
                        help="The id of the experiment of stage1.")
    parser.add_argument("--path_data",
                        type=str,
                        default="datasets/YCB_Video_Dataset",
                        help="The path to YCB-Video dataset.")
    parser.add_argument("--refiner",
                        type=str,
                        default="refiner_MLP",
                        help="The name of the refiner.")
    parser.add_argument("--iteration",
                        type=int,
                        default=2,
                        help="The number of iterations for pose refinement.")
    args_cfg = parser.parse_args()

    return args_cfg

def init():
    args         = get_parser()

    cfg          = gorilla.Config.fromfile(args.config)
    exp_name     = "refiner_" + args.refiner + '_' + osp.splitext(args.config.split("/")[-1])[0] + '_id' + str(args.exp_id) + '_model_' + args.model + '_' + osp.splitext(args.config_stage1.split("/")[-1])[0] + '_id' + str(args.exp_id_stage1) + "_epoch_" + str(args.epoch_stage1)
    log_dir       = osp.join("log", exp_name)
    exp_name_model= args.model + '_' + osp.splitext(args.config_stage1.split("/")[-1])[0] + '_id' + str(args.exp_id_stage1)
    log_dir_model = osp.join("log", exp_name_model)
    cfg.exp_name = exp_name
    cfg.gpus     = args.gpus
    cfg.model_name  = args.model
    cfg.refiner_name= args.refiner
    cfg.log_dir    = log_dir
    cfg.log_dir_model = log_dir_model
    cfg.test_epoch = args.epoch
    cfg.test_epoch_stage1 = args.epoch_stage1
    cfg.path_data    = args.path_data
    cfg.iteration    = args.iteration

    tools_train.backup(["tools/test_YCBV_stage2.py"], log_dir)
    gorilla.utils.set_cuda_visible_devices(gpu_ids = cfg.gpus)
    logger = tools_train.get_logger(level_print=logging.INFO, level_save=logging.WARNING, path_file=log_dir+"/test_epoch_" + str(cfg.test_epoch)+"_iter_"+str(cfg.iteration)  + "_logger.log")

    return logger, cfg


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

def VOCap(rec, prec):
    idx = np.where(rec != np.inf)
    if len(idx[0]) == 0:
        return 0
    rec = rec[idx]
    prec = prec[idx]
    mrec = np.array([0.0]+list(rec)+[0.1])
    mpre = np.array([0.0]+list(prec)+[prec[-1]])
    for i in range(1, prec.shape[0]):
        mpre[i] = max(mpre[i], mpre[i-1])
    i = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i-1]) * mpre[i]) * 10
    return ap

def cal_dis_acc(dis_array, threshold):
    n = dis_array.shape[0]
    c = (dis_array<threshold).sum()
    acc = c/n
    return acc
def cal_auc_acc(dis_list, max_dis=0.1):
    D = np.array(dis_list)
    D[np.where(D > max_dis)] = np.inf
    D = np.sort(D)
    n = len(dis_list)
    acc = np.cumsum(np.ones((1,n)), dtype=np.float32) / n
    aps = VOCap(D, acc)
    acc_2cm = cal_dis_acc(D, threshold=0.02)
    return aps * 100, acc_2cm * 100
def cal_metric_auc_acc(ADDS_list, idx_list, logger):
    ADDS_list = np.array(ADDS_list)
    idx_list = np.array(idx_list)
    ADDS_auc_list = []
    ADDS_acc_list  = []
    for idx in range(21):
        ADDS_list_item = ADDS_list[np.where(idx_list==idx)]
        ADDS_auc_item, acc_item = cal_auc_acc(ADDS_list_item)
        ADDS_acc_list.append(acc_item)
        ADDS_auc_list.append(ADDS_auc_item)
        logger.warning('NO.{0} | ADDS_AUC:{1} | ADDS<2cm:{2}'.format('%02d'%(idx+1), '%3.2f'%ADDS_auc_item, '%3.2f'%acc_item))
    ADDS_auc_mean = round(np.mean(ADDS_auc_list), 2)
    acc_mean       = round(np.mean(ADDS_acc_list), 2)
    logger.warning('MEAN  | ADDS_AUC:{0} | ACC<2cm:{1}'.format('%3.2f'%ADDS_auc_mean, '%3.2f'%acc_mean))
    return  ADDS_auc_mean

def test(model, refiner, iteration, cfg, logger):

    model.eval()
    refiner.eval()
    sym_list = [12, 15, 18, 19, 20]

    dataset = Dataset.YCBDataset('eval', cfg.hyper_dataset_test, root=cfg.path_data)
    dataloder = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.hyper_dataloader_test.bs,
            collate_fn=dataset.collate,
            num_workers=cfg.hyper_dataloader_test.num_workers,
            shuffle=cfg.hyper_dataloader_test.shuffle,
            sampler=None,
            drop_last=cfg.hyper_dataloader_test.drop_last,
            pin_memory=cfg.hyper_dataloader_test.pin_memory
        )

    ADDS_list = []
    idx_list = []

    class_file = open('./YCBV/utils_YCBV/classes.txt')
    class_id = 1
    cld = []
    while 1:
        class_input = class_file.readline()
        if not class_input:
            break
        class_input = class_input[:-1]

        input_file = open('{0}/models/{1}/points.xyz'.format(cfg.path_data, class_input))
        cld_tmp = []
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            input_line = input_line[:-1].split(' ')
            cld_tmp.append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
        cld_tmp = np.array(cld_tmp)[0:2620, :]
        cld.append(cld_tmp)
        input_file.close()

        class_id += 1
    cld = torch.FloatTensor(np.stack(cld)).cuda()
    print(cld.size())

    with tqdm(total=len(dataloder), ncols=80) as t:
        for j, data in enumerate(dataloder):

            all_flags = data["all_flags"]
            cls_label = data["obj_idx"]
            with torch.no_grad():
                outputs_main     = model(data)
                rot_cur          = outputs_main["rot_pred"]
                trans_cur        = outputs_main["trans_pred"]
                points_inp       = data["labels"]['points_inp'].cuda()
                cur_cld          = cld[cls_label.long().cuda()[all_flags.cuda()==1]]
                points_inp_cur           = torch.bmm(points_inp-trans_cur.unsqueeze(1), rot_cur)
                point_feats_inp_RE_embed = outputs_main["point_feats_inp_RE_embed"]
                inp_refiner              = torch.cat([points_inp_cur.transpose(1,2), point_feats_inp_RE_embed], dim = 1).detach()
                conf                     = outputs_main["conf"]
                for i in range(iteration):
                    dict_inp_refiner = {}
                    dict_inp_refiner["input_features"] = inp_refiner
                    dict_inp_refiner["conf"]           = conf
                    dict_inp_refiner["obj_idx"]        = cls_label
                    outputs_refiner = refiner(dict_inp_refiner)
                    rot_ref     = outputs_refiner["rot_pred"]
                    trans_ref   = outputs_refiner["trans_pred"]
                    trans_cur   = (rot_cur @ trans_ref.unsqueeze(2)).squeeze(2) + trans_cur
                    rot_cur     = rot_cur @ rot_ref
                    points_inp_cur = torch.bmm(points_inp-trans_cur.unsqueeze(1), rot_cur)
                    inp_refiner    = torch.cat([points_inp_cur.transpose(1,2), point_feats_inp_RE_embed], dim = 1).detach()

            points_tmp = data["labels"]["points_tmp"]
            rot_pred   = rot_cur
            trans_pred = trans_cur
            rot_gt     = data["labels"]["rot_gt"].cuda()
            trans_gt   = data["labels"]["trans_gt"].cuda()

            cur_cld = cld[cls_label.long().cuda()[all_flags.cuda()==1]]
            points_tmp_posed_pred = torch.bmm(cur_cld, rot_pred.transpose(1,2)) + trans_pred.unsqueeze(1)
            points_tmp_posed_gt   = torch.bmm(cur_cld, rot_gt.transpose(1,2))   + trans_gt.unsqueeze(1)
            cd_dis = torch.mean(torch.min(torch.norm(points_tmp_posed_pred.unsqueeze(2) - points_tmp_posed_gt.unsqueeze(1), dim=3), 2)[0], dim=1)
            
            batch_id = 0
            for k in range(all_flags.size(0)):
                if all_flags[k].item() == 0: 
                    ADDS_list.append(np.Inf)
                    # print("---------------------------------------------Inf---------------------------------------------")
                else:
                    ADDS_list.append(cd_dis[batch_id].item())

                    batch_id += 1
                idx_list.append(cls_label[k].item())
            assert batch_id == points_tmp.size(0)
            t.set_description(
                "Test [{}/{}]".format(j+1, len(dataloder))
            )
            t.update(1)
    ADDS_auc_mean = cal_metric_auc_acc(ADDS_list, idx_list, logger)




if __name__ == "__main__":
    logger, cfg = init()

    logger.warning("************************ Start Logging ************************")
    logger.info(cfg)
    logger.info("using gpu: {}".format(cfg.gpus))

    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)


    # model
    BASE_DIR = os.getcwd()
    ROOT_DIR = BASE_DIR
    sys.path.append(os.path.join(ROOT_DIR, 'YCBV'))
    sys.path.append(os.path.join(ROOT_DIR, 'models'))
    sys.path.append(os.path.join(ROOT_DIR, 'YCBV', 'utils_YCBV'))
    Dataset = importlib.import_module(cfg.hyper_dataset_test.name)
    logger.info("=> creating model ...")
    MODEL = importlib.import_module(cfg.model_name)
    model = MODEL.Network(cfg=cfg.model, mode='test')
    REFINER = importlib.import_module(cfg.refiner_name)
    refiner = REFINER.Refiner(cfg=cfg.model)
    if len(cfg.gpus)>1:
        model = torch.nn.DataParallel(model, range(len(cfg.gpus.split(","))))
    model   = model.cuda()
    refiner = refiner.cuda()
    checkpoint_model   = os.path.join(cfg.log_dir_model, 'epoch_' + str(cfg.test_epoch_stage1) + '.pth')
    checkpoint_refiner = os.path.join(cfg.log_dir, 'epoch_' + str(cfg.test_epoch) + '.pth')
    gorilla.solver.load_checkpoint(model=model, filename=checkpoint_model)
    gorilla.solver.load_checkpoint(model=refiner, filename=checkpoint_refiner)

    test(model, refiner, cfg.iteration, cfg, logger)
