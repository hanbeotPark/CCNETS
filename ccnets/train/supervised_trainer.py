'''
    COPYRIGHT (c) 2022. CCNets. All Rights reserved.
'''
from ..utils.func import get_loss
from .trainer_base import TrainerBase  
import torch

class SupervisedTrainer(TrainerBase):
    def __init__(self, args, _models):
        super(SupervisedTrainer, self).__init__(args, _models)

    def train_models(self, x, y):
        self.set_train(train = True)
        supervised_learning_model = self.models[0]
        
        ################################  Forward Pass  ################################################
        inferred_y = supervised_learning_model(x)
        
        ################################  Prediction Losses  ###########################################
        inference_loss = self.loss_fn(inferred_y, y)

        ################################  Backward Pass  ###############################################
        inference_loss.backward()
        
        ################################  Model Update  ###############################################
        self.update_optimizers()    
        self.update_schedulers()
        ###############################################################################################
        loss = get_loss(inference_loss)
        return loss
    # This part was added after looking at the corresponding code.
    def test_models(self, x, y):
        self.set_train(train = False)
        supervised_learning_model = self.models[0]
        
        ################################  Forward Pass  ################################################
        inferred_y = supervised_learning_model(x)
        
        ################################  Prediction Losses  ###########################################
        inference_loss = self.loss_fn(inferred_y, y)
        
        loss = get_loss(inference_loss)
        return loss