from tensorboardX import SummaryWriter
import os
import logging
from shutil import copyfile


class tools_writer():
    def __init__(self, dir_project, num_counter, get_sum, start_point=0):
        if not os.path.isdir(dir_project):
            os.makedirs(dir_project)
        if get_sum:
            writer = SummaryWriter(dir_project)
        else:
            writer = None
        self.writer = writer
        self.num_counter = num_counter
        self.list_couter = []
        for i in range(num_counter):
            self.list_couter.append(start_point)

    
    def update_scalar(self, list_name, list_value, index_counter, prefix):
        for name, value in zip(list_name, list_value):
            self.writer.add_scalar(prefix+name, float(value), self.list_couter[index_counter])
        
        self.list_couter[index_counter] += 1

    def refresh(self):
        for i in range(self.num_counter):
            self.list_couter[i] = 0

def get_logger(level_print, level_save, path_file, name_logger = "logger"):
    logger = logging.getLogger(name_logger)
    logger.setLevel(level = logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # set file handler
    handler_file = logging.FileHandler(path_file)
    handler_file.setLevel(level_save)
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_file)
    # set console holder
    handler_view = logging.StreamHandler()
    handler_view.setFormatter(formatter)
    handler_view.setLevel(level_print)
    logger.addHandler(handler_view)
    return logger


def debug_params(model):
    for name, params in model.named_parameters():
        print(name, params.sum())
def compare_two_models(model_1, model_2):
    dict_model_1 = {}
    for name, params in model_1.named_parameters():
        dict_model_1[name] = params.sum()
    for name, params in model_2.named_parameters():
        if not (params.sum()==dict_model_1[name]):
            print("ERROR DIFF in: ", name, ":", params.sum()-dict_model_1[name])

def backup(paths, dir_dest):
    for path in paths:
        name = path.split("/")[-1]
        path_save = os.path.join(dir_dest, name)
        copyfile(path, path_save)
        print("Warning, saving to:", path_save)