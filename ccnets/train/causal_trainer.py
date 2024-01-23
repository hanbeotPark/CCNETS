'''
    Causal learning implementation in PyTorch.
    Author:
        PARK, JunHo, junho.zerotoone@gmail.com

    COPYRIGHT (c) 2022. CCNets. All Rights reserved.
'''
from ..utils.func import get_losses, get_errors
from .trainer_base import TrainerBase  
import torch

class CausalTrainer(TrainerBase):
    def __init__(self, args, _models):
        super(CausalTrainer, self).__init__(args, _models)
        self.init_lr = args.lr

    def train_models(self, x, y):
        self.set_train(train = True)
        explainer, reasoner, producer = self.models

        ################################  Forward Pass  ################################################
        e = explainer(x)

        inferred_y = reasoner(x, e)

        generated_x = producer(y, e)
        
        reconstructed_x = producer(inferred_y, e.detach())
        
        ################################  Prediction Losses  ###########################################
        inference_loss = self.loss_fn(reconstructed_x, generated_x.detach())
        generation_loss = self.loss_fn(generated_x, x)
        reconstruction_loss = self.loss_fn(reconstructed_x, x)

        ################################  Model Losses  ################################################
        explainer_error = self.error_fn(inference_loss + generation_loss, reconstruction_loss.detach())
        reasoner_error = self.error_fn(reconstruction_loss + inference_loss, generation_loss.detach())
        producer_error = self.error_fn(generation_loss + reconstruction_loss, inference_loss.detach())
        
        ################################  Backward Pass  ###############################################
        self.backwards(
            [explainer_error, reasoner_error, producer_error],
            [explainer, reasoner, producer])
        ################################  Model Update  ###############################################
        self.update_optimizers()    
        self.update_schedulers()
        self.update_seed()
        
        losses = get_losses([inference_loss, generation_loss, reconstruction_loss])
        errors = get_errors([explainer_error, reasoner_error, producer_error])
        return losses, errors
    
    # This part was added after looking at the corresponding code.
    def test_models(self, x, y):
        self.set_train(train = False)
        explainer, reasoner, producer = self.models

        ################################  Forward Pass  ################################################
        e = explainer(x)

        inferred_y = reasoner(x, e)
        
        generated_x = producer(y, e)
        
        reconstructed_x = producer(inferred_y, e.detach())
        
        
        ################################  Prediction Losses  ###########################################
        inference_loss = self.loss_fn(reconstructed_x, generated_x.detach())
        generation_loss = self.loss_fn(generated_x, x)
        reconstruction_loss = self.loss_fn(reconstructed_x, x)

        ################################  Model Losses  ################################################
        explainer_error = self.error_fn(inference_loss + generation_loss, reconstruction_loss.detach())
        reasoner_error = self.error_fn(reconstruction_loss + inference_loss, generation_loss.detach())
        producer_error = self.error_fn(generation_loss + reconstruction_loss, inference_loss.detach())
        
        ################################  Backward Pass  ###############################################

        
        losses = get_losses([inference_loss, generation_loss, reconstruction_loss])
        errors = get_errors([explainer_error, reasoner_error, producer_error])
        return losses, errors
    
    