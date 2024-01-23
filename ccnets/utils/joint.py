'''
COPYRIGHT (c) 2022. CCNets. All Rights reserved.
'''
import torch
import random
import numpy as np
import torch.nn.functional as F
from torch import nn
from .func import create_layer


class JointLayer(nn.Module):
    """Base class for creating joint layers."""
    def __init__(self, input_dim1, input_dim2, output_dim, act):
        super(JointLayer, self).__init__()
        self.layer_a = create_layer(input_dim1, output_dim, last_act = act)
        self.layer_b = create_layer(input_dim2, output_dim, last_act = act)

    @staticmethod
    def create(inp_size1, inp_size2, outp_size, joint_type, act):
        if joint_type == "none":
            return NoneJointLayer(inp_size1, inp_size2, outp_size, act)
        elif joint_type == "cat":
            return CatJointLayer(inp_size1, inp_size2, outp_size, act)
        elif joint_type == "add":
            return AddJointLayer(inp_size1, inp_size2, outp_size, act)

    def forward(self, tensor_a, tensor_b):
        """To be implemented by subclasses."""
        pass

class NoneJointLayer(JointLayer):
    def forward(self, tensor_a, tensor_b):
        joint_representation = (self.layer_a(tensor_a), self.layer_b(tensor_b))
        return joint_representation
    
class CatJointLayer(nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim, act):
        #super().__init__(input_dim1, input_dim2)
        super(CatJointLayer, self).__init__() 
        self.cat_emb  = create_layer(input_dim1 + input_dim2, output_dim, last_act=act) 

    """A joint layer that adds the outputs of two linear layers."""
    def forward(self, tensor_a, tensor_b):
        z = torch.cat([tensor_a, tensor_b], dim = -1)
        joint_representation = self.cat_emb(z)
        return joint_representation

class AddJointLayer(JointLayer):
    """A joint layer that adds the outputs of two linear layers."""
    def forward(self, tensor_a, tensor_b):
        joint_representation = (self.layer_a(tensor_a) + self.layer_b(tensor_b))/2.
        return joint_representation
