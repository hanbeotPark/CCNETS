'''
COPYRIGHT (c) 2022. CCNets. All Rights reserved.
'''
import torch
import random
import numpy as np

def set_random_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

def set_dim_args(self, args):
    self.batch_size = args.batch_size
    self.obs_size = args.obs_size
    self.label_size = args.label_size
    self.explain_size = args.explain_size
    self.seq_len = args.seq_len

def set_general_args(self, args):
    self.device = args.device
    self.num_workers = args.num_workers
    self.num_epoch = args.num_epoch
    self.ngpu = args.ngpu
    self.log = args.log if hasattr(args, "log") else None
        
    self.model_path = args.model_path
    self.temp_path = args.temp_path
    self.image_path = args.image_path

    self.label_type = args.label_type

    self.loss_type = args.loss_type
    self.trainer_type = args.trainer_type 
    self.selected_indices = args.selected_indices
    self.num_checkpoints = int(args.num_checkpoints)
    
def set_flag_args(self, args):
    self.use_one_hot = args.use_one_hot
    self.use_test = args.use_test
    self.use_shuffle = args.use_shuffle
    self.use_image = args.use_image
    self.cnt_save_image = args.start_cnt_save_image
    return

def set_train_param_args(self, args):
    self.device = args.device
    self.num_workers = args.num_workers

    self.ngpu = args.ngpu
    self.lr = args.lr
    self.gamma = args.gamma
    self.step_size = args.step_size
    return

