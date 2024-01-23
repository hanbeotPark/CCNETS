'''
    COPYRIGHT (c) 2022. CCNets. All Rights reserved.
'''

import torch
import time

from .utils.loader import get_Dataloader, get_testloader
from .utils.setting import set_general_args, set_dim_args, set_flag_args
from .utils.log import log_train, log_metric, log_test
from .utils.print import print_iter, print_train, print_lr, print_metric
from .utils.report import get_classification_report, get_regression_report, get_roc_auc_score
from .utils.loader import _load_model, _save_model
from .utils.func import select_seq_batch, convert_to_one_hot_vector

from tqdm.notebook import tqdm_notebook
from .comps.cooperative_net import CooperativeNet
from .train.causal_trainer import CausalTrainer
from torch.utils.data import Dataset
import numpy as np 
import matplotlib.pyplot as plt

class CCNets(CooperativeNet):
    def __init__(self, args, explainer_net, reasoner_net, producer_net):
        set_dim_args(self, args)
        set_flag_args(self, args)
        set_general_args(self, args)
        
        super(CCNets, self).__init__(args, explainer_net, reasoner_net, producer_net)
        self.models = self.get_models()
        self.trainer = CausalTrainer(args, self.models) 
        self.optimizers = self.trainer.get_optimizers()
        self.schedulers = self.trainer.get_schedulers()
        self.array_explains= None
    def save_models(self, model_path = None):
        model_path = self.model_path if model_path is None else model_path
        for info in zip(self.model_names, self.models, self.optimizers, self.schedulers):
            _names, _model, _optimizers, _schedulers = info
            _save_model(model_path, _names, _model, _optimizers, _schedulers)

    def load_models(self, model_path = None):
        model_path = self.model_path if model_path is None else model_path
        for info in zip(self.model_names, self.models, self.optimizers, self.schedulers):
            _names, _model, _optimizers, _schedulers = info
            _load_model(model_path, _names, _model, _optimizers, _schedulers)
        return self.models

    def init_batch(self, source_batch, target_batch):
        if self.use_one_hot:
            target_batch = convert_to_one_hot_vector(target_batch, self.label_size)

        if self.seq_len > 0:
            source_batch, target_batch = select_seq_batch(source_batch, target_batch, self.seq_len)

        if self.seq_len == 0 and len(target_batch.shape) == 1:
            target_batch.unsqueeze_(-1)                
        elif self.seq_len > 0 and len(target_batch.shape) == 2:
            target_batch.unsqueeze_(-1)                

        source_batch = source_batch.float().to(self.device)
        target_batch = target_batch.float().to(self.device)
        return source_batch, target_batch 

    def test(self, testset):
        self.trainer.set_train(train = False)
        batch_size = int(min(len(testset), self.batch_size))
        dataloader = get_testloader(testset, batch_size)
        array_labels, array_preds = None, None 
        array_generate, array_reconstruct = None, None
        metrics = None
        self.test_iters =0;self.test_cnt_checkpoints=0;self.test_sum_errors=None;self.test_sum_losses=None
        for i, (source_batch, target_batch) in enumerate(dataloader):
                
            source_batch, target_batch = self.init_batch(source_batch, target_batch)
            inferred_y = self.infer(source_batch)
                
            if self.seq_len > 0:
                inferred_y = inferred_y[:,-1,:]
                target_batch = target_batch[:,-1,:]

            if array_preds is None:
                array_preds = inferred_y.cpu()
                array_labels = target_batch.cpu()
            
            else:
                array_preds = torch.cat([array_preds, inferred_y.cpu()], dim = 0)
                array_labels = torch.cat([array_labels, target_batch.cpu()], dim = 0)
                
            if len(source_batch) != batch_size:
                    continue
            losses, errors = self.trainer.test_models(source_batch, target_batch)
            self.test_iters += 1; self.test_cnt_checkpoints += 1
            self.test_sum_losses = (self.test_sum_losses + losses) if self.test_sum_losses is not None else losses
            self.test_sum_errors = (self.test_sum_errors + errors) if self.test_sum_errors is not None else errors
        

        if self.label_type == "UC":
            metrics = get_classification_report(array_preds, array_labels, self.label_size)
        elif self.label_type == "C":
            metrics = get_roc_auc_score(array_preds, array_labels)
        elif self.label_type == "R":
            metrics = get_regression_report(array_preds, array_labels)
        return metrics
    
    def train(self, trainset:Dataset, testset:Dataset = None):
        self.init_training_vars(trainset)
        batch_size = int(min(len(trainset), self.batch_size))
        for epoch in tqdm_notebook(range(self.num_epoch)):
            dataloader = get_Dataloader(trainset, batch_size, self.use_shuffle)
            
            for i, (source_batch, target_batch) in enumerate(dataloader):
                if len(source_batch) != batch_size:
                    continue
                
                source_batch, target_batch = self.init_batch(source_batch, target_batch)
                
                losses, errors = self.trainer.train_models(source_batch, target_batch)
                self.iters += 1; self.cnt_checkpoints += 1
                self.sum_losses = (self.sum_losses + losses) if self.sum_losses is not None else losses
                self.sum_errors = (self.sum_errors + errors) if self.sum_errors is not None else errors

                if self.should_checkpoint():
                    self.process_checkpoint(epoch, i, len(dataloader), testset)

        return
    
    def init_training_vars(self, dataset):
        self.sum_losses, self.sum_errors = None, None
        self.iters, self.cnt_checkpoints, self.cnt_print = 0, 0, 0
        self.pvt_time = time.time()
        self.image_debugger = ImageDebugger(dataset, self.selected_indices, self.obs_size, self.label_size, self.device) if self.use_image else None
        return 
    
    def should_checkpoint(self):
        return self.cnt_checkpoints >= self.num_checkpoints

    def time_checkpoint(self):
        cur_time = time.time()
        et = cur_time - self.pvt_time
        self.pvt_time = cur_time
        return et
    
    def process_checkpoint(self, epoch, i, len_dataloader, testset):
        if self.use_image:
            self.image_debugger.update_images(self.models)
            self.image_debugger.display_image()
            
        et = self.time_checkpoint()
        mean_losses, mean_errors = self.sum_losses/self.cnt_checkpoints, self.sum_errors/self.cnt_checkpoints
        print_iter(epoch, self.num_epoch, i, len_dataloader, et)
        print_lr(self.optimizers)
        print_train(mean_losses, mean_errors)

        if self.use_test:
            matric = self.test(testset) 
            print_metric(self.label_type, matric)
            log_metric(self.log, self.label_type, epoch, matric)
            test_mean_erros = self.test_sum_errors/self.test_cnt_checkpoints
            test_mean_losses = self.test_sum_losses/self.test_cnt_checkpoints
            log_test(self.log, epoch,test_mean_losses,test_mean_erros)
        log_train(self.log,  epoch, mean_losses, mean_errors)
        
        self.log.flush()
        
        save_path = self.model_path if self.cnt_print %2 == 0 else self.temp_path
        self.save_models(model_path = save_path)
        
        self.sum_losses, self.sum_errors, self.cnt_checkpoints = None, None, 0
        self.cnt_print += 1