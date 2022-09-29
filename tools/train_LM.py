import open3d as o3d
import utils.tools_train as tools_train
import gorilla
import argparse
import os
import sys
import os.path as osp
import time
import logging
import numpy as np
import random
import importlib

import torch

BASE_DIR = os.getcwd()
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "models"))
sys.path.append(os.path.join(ROOT_DIR, 'LM'))

def get_parser():
    parser = argparse.ArgumentParser(
        description="LM train")

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
                        default="datasets/Linemod_preprocessed",
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
    cfg.path_data = args.path_data

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    logger = tools_train.get_logger(level_print=logging.INFO, level_save=logging.WARNING, path_file=log_dir+"/training_logger.log")

    tools_train.backup(["tools/train_LM.py", "models/Modules.py", "models/"+args.model+".py", "LM/"+cfg.hyper_dataset_train.name+".py", args.config], log_dir)
    gorilla.utils.set_cuda_visible_devices(gpu_ids = cfg.gpus)

    return logger, cfg

class Trainer(gorilla.solver.BaseSolver):
    def __init__(self, model, loss, dataloaders, logger, cfg):
        super(Trainer, self).__init__(
            model = model,
            dataloaders = dataloaders,
            cfg = cfg,
            logger = logger,
        )
        self.loss = loss
        self.logger = logger
        self.logger.propagate = 0
        tb_writer_ = tools_train.tools_writer(dir_project = cfg.log_dir, num_counter = 2, get_sum=False)
        tb_writer_.writer = self.tb_writer
        self.tb_writer  = tb_writer_

        self.per_val    = cfg.per_val
        self.per_write  = cfg.per_write
        self.epoch = 1

    def solve(self):
        while self.epoch<=self.cfg.max_epoch:
            self.logger.info('\nEpoch {} :'.format(self.epoch))

            end = time.time()
            dict_info_train = self.train()
            train_time = time.time()-end

            dict_info = {'train_time(min)': train_time/60.0}
            for key, value in dict_info_train.items():
                if 'loss' in key:
                    dict_info['train_'+key] = value

            if self.epoch % 10 ==0:
                ckpt_path = os.path.join(cfg.log_dir, 'epoch_' + str(self.epoch) + '.pth')
                gorilla.solver.save_checkpoint(model=self.model, filename=ckpt_path, optimizer=self.optimizer, scheduler=self.lr_scheduler, meta={})

            prefix = 'Epoch {} - '.format(self.epoch)
            write_info = self.get_logger_info(prefix, dict_info=dict_info)
            self.logger.warning(write_info)
            self.epoch += 1
    
    def train(self):
        mode = 'train'
        self.model.train()
        end = time.time()

        for i, data in enumerate(self.dataloaders["train"]):
            data_time = time.time()-end

            self.optimizer.zero_grad()
            loss, dict_info_step = self.step(data, mode)
            forward_time = time.time()-end-data_time

            loss.backward()
            self.optimizer.step()
            backward_time = time.time()- end -forward_time-data_time

            dict_info_step.update({
                'T_data': data_time,
                'T_forward': forward_time,
                'T_backward': backward_time,
            })
            self.log_buffer.update(dict_info_step)

            if i % self.per_write == 0:
                self.log_buffer.average(self.per_write)
                prefix = '[{}/{}][{}/{}] Train - \n'.format(self.epoch, self.cfg.max_epoch, i, len(self.dataloaders["train"]))
                write_info = self.get_logger_info(prefix, dict_info=self.log_buffer._output)
                self.logger.info(write_info)
                self.write_summary(self.log_buffer._output, mode)
            end = time.time()

        dict_info_epoch = self.log_buffer.avg
        self.log_buffer.clear()
        self.lr_scheduler.step()

        return dict_info_epoch

    def evaluate(self):
        mode = 'eval'
        self.model.eval()

        for i, data in enumerate(self.dataloaders["eval"]):
            with torch.no_grad():
                _, dict_info_step = self.step(data, mode)
                self.log_buffer.update(dict_info_step)
                if i % self.per_write == 0:
                    self.log_buffer.average(self.per_write)
                    prefix = '[{}/{}][{}/{}] Test - '.format(self.epoch, self.cfg.max_epoch, i, len(self.dataloaders["eval"]))
                    write_info = self.get_logger_info(prefix, dict_info=self.log_buffer._output)
                    self.logger.info(write_info)
                    self.write_summary(self.log_buffer._output, mode)
        dict_info_epoch = self.log_buffer.avg
        self.log_buffer.clear()

        return dict_info_epoch

    def step(self, data, mode):
        torch.cuda.synchronize()
        data["labels"]["epoch"] = self.epoch
        outputs = self.model(data)
        labels  = data["labels"]
        dict_losses = self.loss(outputs, labels)

        keys = list(dict_losses.keys())
        dict_info = {'loss_all': 0}
        if 'loss_all' in keys:
            loss_all = dict_losses['loss_all']
            for key in keys:
                dict_info[key] = float(dict_losses[key].item())
        else:
            loss_all = 0
            for key in keys:
                loss_all += dict_losses[key]
                dict_info[key] = float(dict_losses[key].item())
            dict_info['loss_all'] = float(loss_all.item())

        if mode == 'train':
            dict_info['lr'] = self.lr_scheduler.get_lr()[0]

        return loss_all, dict_info

    def get_logger_info(self, prefix, dict_info):
        info = prefix
        for key, value in dict_info.items():
            if 'T_' in key:
                info = info + '{}: {:.3f}\t'.format(key, value)
            else:
                info = info + '{}: {:.5f}\t'.format(key, value)

        return info

    def write_summary(self, dict_info, mode):
        keys   = list(dict_info.keys())
        values = list(dict_info.values())
        if mode == "train":
            self.tb_writer.update_scalar(list_name=keys, list_value=values, index_counter=0, prefix="train_")
        elif mode == "eval":
            self.tb_writer.update_scalar(list_name=keys, list_value=values, index_counter=1, prefix="eval_")
        else:
            assert False


if __name__ == "__main__":
    logger, cfg = init()

    logger.warning("************************ Start Logging ************************")
    logger.info(cfg)
    logger.info("using gpu: {}".format(cfg.gpus))

    random.seed(cfg.rd_seed)
    np.random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)
    def worker_init_fn(worker_id, rd):
        seed = cfg.rd_seed
        seed += worker_id
        np.random.seed(seed)
        print(worker_id)

    # model
    logger.info("=> creating model ...")
    MODEL = importlib.import_module(cfg.model_name)
    model = MODEL.Network(cfg = cfg.model)
    if len(cfg.gpus)>1:
        model = torch.nn.DataParallel(model, range(len(cfg.gpus.split(","))))
    model = model.cuda()
    loss  = MODEL.losses(cfg = cfg.loss).cuda()
    count_parameters = sum(gorilla.parameter_count(model).values())
    logger.warning("#Total parameters : {}".format(count_parameters))

    # dataloader
    train_dataset = importlib.import_module(cfg.hyper_dataset_train.name)
    train_dataset = train_dataset.Dataset('train', cfg.hyper_dataset_train, root=cfg.path_data)

    train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.hyper_dataloader_train.bs,
            collate_fn=train_dataset.collate,
            num_workers=cfg.hyper_dataloader_train.num_workers,
            shuffle=cfg.hyper_dataloader_train.shuffle,
            sampler=None,
            drop_last=cfg.hyper_dataloader_train.drop_last,
            pin_memory=cfg.hyper_dataloader_train.pin_memory,
        )

    dataloaders = {
        "train": train_dataloader,
    }

    trainer = Trainer(model=model, loss=loss, dataloaders=dataloaders, logger=logger, cfg=cfg)
    trainer.solve()

    logger.info('\nFinish!\n')
