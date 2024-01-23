'''
    CCNet's configurators implementations.

    Author:
        PARK, JunHo, junho.zerotoone@gmail.com
'''
import torch.nn as nn
import torch
from ..utils.func import init_weights, create_layer
from torch.distributions import Normal

class Explainer(nn.Module):
    def __init__(self, args, net):
        super(Explainer, self).__init__()
        if args.use_image:
            self.obs_embed = create_layer() 
        else:
            self.obs_embed = create_layer(args.obs_size, args.hidden_size, last_act = "tanh") 
            
        self.net = net(args)
        self.final = create_layer(args.hidden_size, args.explain_size, first_act = "relu", last_act = args.explain_fn)
        self.apply(init_weights) 

    @staticmethod
    def create(args, net):
        return Explainer(args, net)
    
    def forward(self, obs):
        obs = self.obs_embed(obs) 
        e = self.net(obs)
        mean = self.final(e)
        return mean
