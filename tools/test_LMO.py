import open3d
import sys
import os
import importlib

import gorilla
from tqdm import tqdm
import argparse

import os.path as osp
import logging
import numpy as np
import random
import yaml

import torch
import utils.tools_train as tools_train

def get_parser():
    parser = argparse.ArgumentParser(
        description="LMO test")

    parser.add_argument("--epoch",
                        type=int,
                        default=400,
                        help="Checkpoint to be tested.")
    
    parser.add_argument("--gpus",
                        type=str,
                        default="0",
                        help="GPUS to be used.")
    parser.add_argument("--model",
                        type=str,
                        default="DCL_Net",
                        help="The name of the model to be used.")
    parser.add_argument("--config",
                        type=str,
                        default="configs/config_LM.yaml",
                        help="Path to the config file")
    parser.add_argument("--exp_id",
                        type=int,
                        default=10000,
                        help="The id of this experiment.")
    parser.add_argument("--path_data",
                        type=str,
                        default="datasets/occlusion_linemod",
                        help="The path to LineMOD dataset.")
    args_cfg = parser.parse_args()

    return args_cfg

def init():
    args         = get_parser()
    exp_name     = args.model + '_' + osp.splitext(args.config.split("/")[-1])[0] + '_id' + str(args.exp_id)
    log_dir       = osp.join("log", exp_name)

    cfg          = gorilla.Config.fromfile(args.config)
    cfg.exp_name = exp_name
    cfg.gpus     = args.gpus
    cfg.model_name = args.model
    cfg.log_dir  = log_dir
    cfg.test_epoch = args.epoch
    cfg.path_data = args.path_data
    
    diameter = []
    meta_file = open('./datasets/Linemod_preprocessed/models/models_info.yml', 'r')
    meta = yaml.load(meta_file)
    cfg.objlist = [1, 5, 6, 8, 9, 10, 11, 12]
    cfg.num_objects = 8
    for obj in cfg.objlist:
        diameter.append(meta[obj]['diameter'] / 1000.0 * 0.1)
    print(diameter)
    cfg.diameter = diameter

    tools_train.backup(["tools/test_LMO.py"], log_dir)
    gorilla.utils.set_cuda_visible_devices(gpu_ids = cfg.gpus)
    logger = tools_train.get_logger(level_print=logging.INFO, level_save=logging.WARNING, path_file=log_dir+"/test_epoch" + str(cfg.test_epoch)  + "_logger.log")

    return logger, cfg


def test(model, save_path, cfg, logger):

    model.eval()

    dataset = DATASET.Dataset('eval', cfg.hyper_dataset_test, root=cfg.path_data)
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

    success_count = [0 for i in range(cfg.num_objects)]
    num_count = [0 for i in range(cfg.num_objects)]
    fw = open('{0}/eval_result_logs.txt'.format(save_path), 'w')
    count = 0
    with tqdm(total=len(dataloder)) as t:
        for i, data in enumerate(dataloder):
            if data['flags'].shape[0]==1 and (data['flags']==-1).sum()==1:
                count += 1
                obj_idx = data['obj_idx']
                fw.write('No.{0} NOT Pass! Lost detection! Following HybridPose, count it on.\n'.format(count))
                
                
                sym_flag = data['flags']
                batch_id = 0
                for k in range(sym_flag.size(0)):
                    count    += 1
                    idx       = obj_idx[batch_id].item()
                    batch_id += 1
                    num_count[idx] += 1
                t.update(1)
                continue
            data["labels"]["epoch"] = 400
            pred = model(data)

            points_tmp = data["labels"]["points_tmp"]
            
            rot_pred   = pred["rot_pred"]
            trans_pred = pred["trans_pred"]
            rot_gt     = data["labels"]["rot_gt"].cuda()
            trans_gt   = data["labels"]["trans_gt"].cuda()
            points_tmp_posed_pred = torch.bmm(points_tmp, rot_pred.transpose(1,2)) + trans_pred.unsqueeze(1)
            points_tmp_posed_gt   = torch.bmm(points_tmp, rot_gt.transpose(1,2))   + trans_gt.unsqueeze(1)
            sym_flag = data['flags']
            obj_idx = data['obj_idx']
            
            l2_dis = torch.mean(torch.norm(points_tmp_posed_pred - points_tmp_posed_gt, dim=2), dim=1)
            cd_dis = torch.mean(torch.min(torch.norm(points_tmp_posed_pred.unsqueeze(2) - points_tmp_posed_gt.unsqueeze(1), dim=3), 2)[0], dim=1)

            batch_id = 0
            for k in range(sym_flag.size(0)):
                count += 1
                if sym_flag[k].item() == -1:
                    fw.write('No.{0} NOT Pass! Lost detection!\n'.format(count))
                    continue
                elif sym_flag[k].item() == 0:
                    dis = l2_dis[batch_id].item()
                elif sym_flag[k].item() == 1:
                    dis = cd_dis[batch_id].item()

                idx = obj_idx[batch_id].item()
                batch_id += 1
                num_count[idx] += 1
                if dis < cfg.diameter[idx]:
                    success_count[idx] += 1
                    fw.write('No.{0} Pass! Distance: {1}  ({2})\n'.format(count, dis, idx))
                else:
                    fw.write('No.{0} NOT Pass! Distance: {1}  ({2})\n'.format(count, dis, idx))

            assert batch_id == points_tmp.size(0)
            # print(success_count)
            t.set_description(
                "Test [{}/{}][{}/{}] - success rate: {}".format(i+1, len(dataloder), count, dataset.__len__(),  float(sum(success_count)) / sum(num_count))
            )
            t.update(1)

    for i in range(cfg.num_objects):
        logger.warning('Object {0} success rate: {1}'.format(cfg.objlist[i], float(success_count[i]) / num_count[i]))
        fw.write('Object {0} success rate: {1}\n'.format(cfg.objlist[i], float(success_count[i]) / num_count[i]))
    logger.warning('ALL success rate: {0}'.format(float(sum(success_count)) / sum(num_count)))
    fw.write('ALL success rate: {0}\n'.format(float(sum(success_count)) / sum(num_count)))
    fw.close()


if __name__ == "__main__":
    logger, cfg = init()

    logger.warning("************************ Start Logging the Evaluation Process On Occlusion-LineMOD Dataset ************************")
    logger.info(cfg)
    logger.info("using gpu: {}".format(cfg.gpus))

    random.seed(cfg.rd_seed)
    np.random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)

    save_path = os.path.join(cfg.log_dir, 'eval_epoch' + str(cfg.test_epoch))
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    BASE_DIR = os.getcwd()
    ROOT_DIR = BASE_DIR
    sys.path.append(os.path.join(ROOT_DIR, 'LM'))
    sys.path.append(os.path.join(ROOT_DIR, 'models'))
    DATASET = importlib.import_module(cfg.hyper_dataset_test.name[1])
    # model
    logger.info("=> creating model ...")
    MODEL = importlib.import_module(cfg.model_name)
    model = MODEL.Network(cfg=cfg.model, mode='test')
    if len(cfg.gpus)>1:
        model = torch.nn.DataParallel(model, range(len(cfg.gpus.split(","))))
    model = model.cuda()
    checkpoint = os.path.join(cfg.log_dir, 'epoch_' + str(cfg.test_epoch) + '.pth')
    gorilla.solver.load_checkpoint(model=model, filename=checkpoint)

    test(model, save_path, cfg, logger)
