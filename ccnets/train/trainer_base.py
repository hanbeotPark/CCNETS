'''
    Causal learning implementation in PyTorch.
    
    Author:
        PARK, JunHo, junho.zerotoone@gmail.com
        Kim, JinSu, wlstnwkwjsrj@naver.com

    COPYRIGHT (c) 2022. CCNets. All Rights reserved.
'''
# from ..utils.dataloader import *
from ..utils.losses import get_loss_function, get_error_function, reduce_dim
from ..utils.func import set_random_seed
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

class TrainerBase:
    def __init__(self, args, _models):
        self.seed_val = 0
        self.use_seq = True if args.seq_len > 0 else False
        
        self.batch_size = args.batch_size
        self.loss_reduction = args.loss_reduction
        self.error_reduction = args.error_reduction
        self.loss_type = args.loss_type
        self.error_type = args.error_type
        _optimizers = []
        _schedulers = []
        
        for idx, model in enumerate(_models):
            lr = args.lr 
            opt = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
            sc = StepLR(optimizer=opt, step_size = args.step_size, gamma = args.gamma)
            _optimizers.append(opt)
            _schedulers.append(sc)

        self.schedulers = _schedulers
        self.optimizers = _optimizers
        self.models = _models
    
    def set_train(self, train: bool):
        for m in self.models:
            m.requires_grad_(True)
            m.zero_grad()
            m.train(train)
        return
    
    def get_schedulers(self):
        return self.schedulers

    def get_optimizers(self):
        return self.optimizers

    def update_schedulers(self, errors = None):
        for idx, sc in enumerate(self.schedulers):
            if errors is None:
                sc.step() 
            else:
                sc.step(errors[idx]) 

    def update_optimizers(self):
        for opt in self.optimizers:
            opt.step()        

    def reset_seed(self):
        set_random_seed(self.seed_val)

    def update_seed(self):
        self.seed_val += 1
        set_random_seed(self.seed_val)

    def loss_fn(self, predict, target):
        if self.use_seq:
            predict = predict[:,-1,:]
            if self.loss_type != "CrossEntropy":
                target = target[:,-1,:]
        loss = get_loss_function(predict, target, self.loss_type)
        loss = reduce_dim(loss, self.loss_reduction)
        return loss 

    def error_fn(self, predict, target):
        error = get_error_function(predict, target, self.error_type)
        error = reduce_dim(error, self.error_reduction)
        return error 

    def backwards(self, errors, models):
        len_model = len(models)
        for idx in range(len_model):
            models[idx].requires_grad_(False)
        for model_idx, model in enumerate(models):
            model.requires_grad_(True)
            for i in range(model_idx, len_model):
                models[i].zero_grad()

            error = errors[model_idx]
            if model_idx < len_model - 1:
                error.backward(torch.ones_like(error), retain_graph = True)
            else:
                error.backward(torch.ones_like(error), retain_graph = False)

            model.requires_grad_(False)
        for idx in range(len_model):
            models[idx].requires_grad_(True)
        return