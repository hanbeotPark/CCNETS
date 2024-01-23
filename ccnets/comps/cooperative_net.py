'''
    CCNet's configurators implementations.

    Author:
        PARK, JunHo, junho.zerotoone@gmail.com
'''
import torch
from .explainer import Explainer
from .reasoner import Reasoner
from .producer import Producer
from ..utils.setting import set_train_param_args

class CooperativeNet():
    def __init__(self, args, explainer_net, reasoner_net, producer_net):
        set_train_param_args(self, args)
        self.model_names = ["explainer", "reasoner", "producer"]
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.ngpu > 0) else "cpu")
        self.ngpu = args.ngpu
        self.explainer =  Explainer.create(args, explainer_net).to(self.device)
        self.reasoner =  Reasoner.create(args, reasoner_net).to(self.device)
        self.producer =  Producer.create(args, producer_net).to(self.device)
        self._models = [self.explainer, self.reasoner, self.producer]

    def get_models(self):
        return self._models

    def explain(self, x):
        with torch.no_grad():
            e = self.explainer(x)
        return e

    def reason(self, x, e):
        with torch.no_grad():
            y = self.reasoner(x, e)
        return y

    def produce(self, y, e):
        with torch.no_grad():
            x = self.producer(y, e)
        return x

    def infer(self, x):
        with torch.no_grad():
            e = self.explainer(x)
            y = self.reasoner(x, e)
        return y

################################################################################
###############      CCNETS DATA RECREATION TYPE     ###########################
################################################################################

    def generate(self, x, y):
        with torch.no_grad():
            e = self.explainer(x)
            x_generated = self.producer(y, e)
        return x_generated, y

    def generate_with_random_label(self, x, y):
        with torch.no_grad():
            e = self.explainer(x)
            y_random = torch.rand_like(y)
            x_generated = self.producer(y_random, e)
        return x_generated, y_random

    def generate_with_random_explain(self, x, y):
        with torch.no_grad():
            e = self.explainer(x)
            e_random = torch.rand_like(e)
            x_generated = self.producer(y, e_random)
        return x_generated, y

    def generate_random(self, x, y):
        with torch.no_grad():
            e = self.explainer(x)
            random_e = torch.rand_like(e)
            y_random = torch.rand_like(y)
            x_random = self.producer(y_random, random_e)
        return x_random, y_random

    def reconstruct(self, x, y):
        with torch.no_grad():
            e = self.explainer(x)
            y_inferred = self.reasoner(x, e)
            x_reconstructed = self.producer(y_inferred, e)
        return x_reconstructed, y

    def reverse_label(self, x, y):
        with torch.no_grad():
            e = self.explainer(x)
            y_reversed = torch.where(y < 0.5, torch.ones_like(y), torch.zeros_like(y))
            x_reversed = self.producer(y_reversed, e)
        return x_reversed, y_reversed

    def balance_label(self, x, y):
        with torch.no_grad():
            e = self.explainer(x)
            
            y_zeros = torch.zeros_like(y)
            y_ones = torch.ones_like(y)
            
            y_balanced = torch.where(torch.rand_like(y) < 0.5, y_zeros, y_ones)
            x_balanced = self.producer(y_balanced, e)
        return x_balanced, y_balanced
    
    
################################################################################
###############      CCNETS DATA RECREATION TYPE     ###########################
################################################################################