'''
    CCNet's configurators implementations.

    Author:
        PARK, JunHo, junho.zerotoone@gmail.com
'''
import torch
import torch.nn as nn
from ..utils.func import init_weights, create_layer
from ..utils.setting import set_train_param_args

class SupervisedLearningModel(nn.Module):
    def __init__(self, args, net):
        super(SupervisedLearningModel, self).__init__()

        set_train_param_args(self, args)
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.ngpu > 0) else "cpu")
        self.ngpu = args.ngpu

        if args.use_image:
            self.obs_embed = create_layer() 
        else:
            self.obs_embed = create_layer(args.obs_size, args.hidden_size, last_act = "tanh") 
            
        self.net = net(args)
        self.final = create_layer(args.hidden_size, args.label_size, first_act = "relu", last_act = args.label_fn)
        self.apply(init_weights) 

    @staticmethod
    def create(args, net):
        return SupervisedLearningModel(args, net)

    def infer(self, x):
        with torch.no_grad():
            y = self.forward(x)
        return y
    
    def forward(self, obs):
        obs = self.obs_embed(obs) 
        e = self.net(obs)
        return self.final(e)