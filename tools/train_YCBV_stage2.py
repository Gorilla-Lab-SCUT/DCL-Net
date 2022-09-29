import open3d
import os
import sys
import time
import torch
import logging
import gorilla
import argparse
import importlib
import numpy as np
import os.path as osp
import utils.tools_train as tools_train


BASE_DIR = os.getcwd()
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "models"))
sys.path.append(os.path.join(ROOT_DIR, 'YCBV'))

def get_parser():
    parser = argparse.ArgumentParser(
        description="YCBV train")

    parser.add_argument("--gpus",
                        type=str,
                        default="0",
                        help="GPUS to be used.")
    parser.add_argument("--model",
                        type=str,
                        default="DCL_Net",
                        help="The name of the model to be used in stage1.")
    parser.add_argument("--config_stage1",
                        type=str,
                        default="configs/config_YCBV_bs32.yaml",
                        help="Path to the config file used when training in stage1.")
    parser.add_argument("--exp_id_stage1",
                        type=int,
                        default=10000,
                        help="The id of the experiment of stage1.")
    parser.add_argument("--epoch_stage1",
                        type=int,
                        default=-1,
                        help="The epoch of stage1 to be used for the training of stage2.")
    parser.add_argument("--config",
                        type=str,
                        default="config/config_YCBV_bs40.yaml",
                        help="Path to the config file.")
    parser.add_argument("--exp_id",
                        type=int,
                        default=10000,
                        help="The id of this experiment.")
    parser.add_argument("--refiner",
                        type=str,
                        default="refiner_MLP",
                        help="The name of the refiner.")
    parser.add_argument("--iteration",
                        type=int,
                        default=2,
                        help="The number of iterations for pose refinement.")
    parser.add_argument("--path_data",
                        type=str,
                        default="datasets/YCB_Video_Dataset",
                        help="The path to YCB-Video dataset.")
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
    cfg.exp_name_model = exp_name_model
    cfg.gpus     = args.gpus
    cfg.model_name   = args.model
    cfg.refiner_name = args.refiner
    cfg.log_dir      = log_dir
    cfg.log_dir_model= log_dir_model
    cfg.path_data    = args.path_data
    cfg.iteration    = args.iteration
    cfg.epoch_stage1  = args.epoch_stage1

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    logger = tools_train.get_logger(level_print=logging.INFO, level_save=logging.WARNING, path_file=log_dir+"/refining_logger.log")

    tools_train.backup(["tools/train_YCBV_stage2.py", "models/Modules.py", "models/"+args.model+".py", "models/"+args.refiner+".py", "YCBV/"+cfg.hyper_dataset_train.name+".py", args.config], log_dir)
    gorilla.utils.set_cuda_visible_devices(gpu_ids = cfg.gpus)

    return logger, cfg

class Trainer(gorilla.solver.BaseSolver):
    def __init__(self, model, model_main, loss, dataloaders, logger, iteration, cfg):
        super(Trainer, self).__init__(
            model = model,
            dataloaders = dataloaders,
            cfg = cfg,
            logger = logger,
        )
        self.loss = loss
        self.logger = logger
        self.logger.propagate = 0
        self.model_main = model_main
        self.model_main.eval()
        self.iteration = iteration

        tb_writer_ = tools_train.tools_writer(dir_project = cfg.log_dir, num_counter = 2, get_sum=False, start_point=0)
        tb_writer_.writer = self.tb_writer
        self.tb_writer  = tb_writer_

        self.per_val    = cfg.per_val
        self.per_write  = cfg.per_write
        self.per_save   = cfg.per_save
        self.epoch      = 1
        self.iteration_train  = 1
        self.grad_clipper = AutoClip(50)
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
        self.cld = torch.FloatTensor(np.stack(cld)).cuda()
        print(self.cld.size())
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

            if self.epoch % self.per_save ==0:
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

            self.grad_clipper(self.model)
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
            self.lr_scheduler.step()
            self.iteration_train += 1
            
        dict_info_epoch = self.log_buffer.avg
        self.log_buffer.clear()

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
        with torch.no_grad():
            outputs_main = self.model_main(data)
        rot_cur          = outputs_main["rot_pred"]
        trans_cur        = outputs_main["trans_pred"]
        conf             = outputs_main["conf"].detach()
        points_inp       = data["labels"]['points_inp'].cuda()
        cls_label        = data["labels"]["obj_idx"].squeeze(-1)
        cur_cld          = self.cld[cls_label.long()].cuda()
        points_inp_cur           = torch.bmm(points_inp-trans_cur.unsqueeze(1), rot_cur)
        F_Xo_p = outputs_main["F_Xo_p"]
        inp_refiner              = torch.cat([points_inp_cur.transpose(1,2), F_Xo_p], dim = 1).detach()
        for i in range(self.iteration):
            dict_inp_refiner= {}
            dict_inp_refiner["input_features"] = inp_refiner
            dict_inp_refiner["conf"]           = conf
            dict_inp_refiner["obj_idx"]        = cls_label
            
            outputs_refiner = self.model(dict_inp_refiner)
            dict_losses = self.loss(outputs_refiner, trans_cur.detach(), rot_cur.detach(), cur_cld, outputs_main["sym_flag"], data["labels"])

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
            loss_all.backward()
            # update 
            rot_ref     = outputs_refiner["rot_pred"]
            trans_ref   = outputs_refiner["trans_pred"]
            trans_cur   = (rot_cur @ trans_ref.unsqueeze(2)).squeeze(2) + trans_cur
            rot_cur     = rot_cur @ rot_ref
            points_inp_cur = torch.bmm(points_inp-trans_cur.unsqueeze(1), rot_cur)
            inp_refiner    = torch.cat([points_inp_cur.transpose(1,2), F_Xo_p], dim = 1).detach()
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

class AutoClip:
    def __init__(self, percentile):
        self.grad_history = []
        self.percentile = percentile
        
    def compute_grad_norm(self, model):
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        return total_norm 

    def __call__(self, model):
        grad_norm = self.compute_grad_norm(model)
        self.grad_history.append(grad_norm)
        clip_value = np.percentile(self.grad_history, self.percentile)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value) 

if __name__ == "__main__":
    logger, cfg = init()

    logger.warning("************************ Start Logging ************************")
    logger.info(cfg)
    logger.info("using gpu: {}".format(cfg.gpus))

    gorilla.set_random_seed(cfg.rd_seed)
    def worker_init_fn(worker_id, rd):
        seed = cfg.rd_seed
        seed += worker_id
        np.random.seed(seed)
        print(worker_id)

    # model
    logger.info("=> creating model ...")
    MODEL = importlib.import_module(cfg.model_name)
    model = MODEL.Network(cfg = cfg.model).cuda()
    REFINER = importlib.import_module(cfg.refiner_name)
    refiner = REFINER.Refiner(cfg = cfg.model).cuda()
    loss    = REFINER.losses_refiner(cfg = cfg.loss).cuda()

    count_parameters = sum(gorilla.parameter_count(model).values())
    logger.warning("#Total parameters of the main model : {}".format(count_parameters))
    count_parameters = sum(gorilla.parameter_count(refiner).values())
    logger.warning("#Total parameters of the refiner    : {}".format(count_parameters))

    # dataloader
    train_dataset = importlib.import_module(cfg.hyper_dataset_train.name)
    train_dataset = train_dataset.Dataset('train', cfg.hyper_dataset_train, root=cfg.path_data)

    train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=int(cfg.hyper_dataloader_train.bs / cfg.iteration),
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

    # solver
    trainer = Trainer(model=refiner, model_main=model, loss=loss, dataloaders=dataloaders, logger=logger, cfg=cfg, iteration=cfg.iteration)
    trainer.lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer = trainer.optimizer, base_lr = cfg.lr_scheduler_cyc.base_lr, max_lr = cfg.lr_scheduler_cyc.max_lr, step_size_up = cfg.lr_scheduler_cyc.step_size_up, step_size_down = cfg.lr_scheduler_cyc.step_size_down, cycle_momentum=False)
    
    checkpoint_model   = os.path.join(cfg.log_dir_model, 'epoch_' + str(cfg.epoch_stage1) + '.pth')
    gorilla.solver.load_checkpoint(model=model, filename=checkpoint_model)
    trainer.solve()

    logger.info('\nFinish!\n')