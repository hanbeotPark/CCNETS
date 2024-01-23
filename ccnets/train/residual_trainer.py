'''
    COPYRIGHT (c) 2022. CCNets. All Rights reserved.
'''
from ..utils.func import get_loss
from .trainer_base import TrainerBase  
import torch

class ResidualTrainer(TrainerBase):
    def __init__(self, args, _models):
        super(ResidualTrainer, self).__init__(args, _models)

    def train_models(self, x, y):
        self.set_train(train = True)
        explainer, reasoner = self.models
        
        ################################  Forward Pass  ################################################
        e = explainer(x)
        inferred_y = reasoner(x, e)
        
        ################################  Prediction Losses  ###########################################
        inference_loss = self.loss_fn(inferred_y, y)

        ################################  Backward Pass  ###############################################
        inference_loss.backward(torch.ones_like(inference_loss))
        
        ################################  Model Update  ###############################################
        self.update_optimizers()    
        self.update_schedulers()
        self.update_seed()
        ###############################################################################################
        loss = get_loss(inference_loss)
        return loss
