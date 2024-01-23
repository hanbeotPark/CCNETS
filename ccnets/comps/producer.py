'''
    CCNet's configurators implementations.

    Author:
        PARK, JunHo, junho.zerotoone@gmail.com
'''
import torch
import torch.nn as nn
from ..utils.func import init_weights, create_layer
from ..utils.joint import JointLayer

class Producer(nn.Module):
    def __init__(self, args, net):
        super(Producer, self).__init__()
        self.joint_type = args.producer_joint_type
        self.use_swap_inputs = args.use_producer_swap_inputs

        inp_size1, inp_size2 = (args.label_size, args.explain_size) if not self.use_swap_inputs else (args.explain_size, args.label_size)
        self.joint_layer = JointLayer.create(inp_size1, inp_size2, args.hidden_size, self.joint_type, act = "tanh")
            
        self.net = net(args)
        self.final = create_layer(args.hidden_size, args.obs_size, first_act = "relu", last_act = args.obs_fn) if not args.use_image else create_layer()
        self.apply(init_weights)

    @staticmethod
    def create(args, net):
        return Producer(args, net)

    def forward(self, label, explain):
        z = self.joint_layer(label, explain) if not self.use_swap_inputs else self.joint_layer(explain, label)

        if self.joint_type == "none":
            x = self.net(*z)
        else:
            x = self.net(z)
        return self.final(x)


