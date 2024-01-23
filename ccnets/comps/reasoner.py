'''
    CCNet's configurators implementations.

    Author:
        PARK, JunHo, junho.zerotoone@gmail.com
'''
import torch
import torch.nn as nn
from ..utils.func import init_weights, multiply_list_elements, create_layer
from ..utils.joint import JointLayer

class Reasoner(nn.Module):
    def __init__(self, args, net):
        super(Reasoner, self).__init__()  # don't forget to call super's init in PyTorch
        hidden_size = args.hidden_size
        self.net = net(args)
        self.final = create_layer(hidden_size, args.label_size, first_act = "relu", last_act = args.label_fn)

    @staticmethod
    def create(args, net):
        use_image = args.use_image
        if use_image:
            return ImageReasoner(args, net)
        else:
            return VectorReasoner(args, net)

    def forward(self, obs, explain):
        pass

class ImageReasoner(Reasoner):
    def __init__(self, args, net):
        super().__init__(args, net)
        self.joint_type = args.reasoner_joint_type
        num_pixel = multiply_list_elements(args.obs_size)
        self.n_mul = num_pixel//args.explain_size
        self.n_rest = num_pixel%args.explain_size

        self.apply(init_weights)
    
    def forward(self, img, explain):
        if self.joint_type == "none":
            z = (img, explain)
        else:
            explain = torch.cat([explain.repeat(1, self.n_mul), torch.zeros_like(explain[:, :self.n_rest])], dim = -1)
            explain = explain.reshape(*img.shape)/self.n_mul
            if self.joint_type == "cat":
                z = torch.cat([img, explain], dim = 1)
            elif self.joint_type == "add":
                z = img + explain
            elif self.joint_type == "mul":
                z = img*explain
        y = self.net(z)
        return self.final(y)            

class VectorReasoner(Reasoner):
    def __init__(self, args, net):
        super().__init__(args, net)
        self.joint_type = args.reasoner_joint_type
        self.use_swap_inputs = args.use_reasoner_swap_inputs

        inp_size1, inp_size2 = (args.obs_size, args.explain_size) if not self.use_swap_inputs else (args.explain_size, args.obs_size)
        self.joint_layer = JointLayer.create(inp_size1, inp_size2, args.hidden_size, self.joint_type, act = "tanh")
        
        self.apply(init_weights)
        
    def forward(self, obs, explain):
        z = self.joint_layer(obs, explain) if not self.use_swap_inputs else self.joint_layer(explain, obs)
        
        if self.joint_type == "none":
            y = self.net(*z)
        else:
            y = self.net(z)

        return self.final(y)
    