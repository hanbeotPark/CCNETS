'''
    DeepFM Variations in PyTorch.

    Author:
        PARK, JunHo, junho.zerotoone@gmail.com
        Kim, JinSu, wlstnwkwjsrj@naver.com

    Reference:
        [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
            Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def create_deep_modules(self, layers_size):
        deep_modules = []
        for in_size, out_size in zip(layers_size[:-1], layers_size[1:]):
            deep_modules.append(nn.Linear(in_size, out_size))
            deep_modules.append(nn.ReLU())
        return nn.Sequential(*deep_modules)

    def __init__(self, args):
        super(MLP, self).__init__()   
        hidden_size = args.hidden_size
        num_layer = args.num_layer
        n_inp = hidden_size
        n_oup = hidden_size
        
        self.output_layer = nn.Linear(int(hidden_size), n_oup)
        
        self.deep = self.create_deep_modules([n_inp] + [int(hidden_size) for i in range(num_layer)])
                
    def forward(self, x):
        x = self.deep(x)
        return self.output_layer(x) 
        
    
class ResBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(ResBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out += residual
        out = self.relu(out)
        return out

class ResMLP(nn.Module):
    def __init__(self, args):
        input_dim, hidden_dim, output_dim, num_blocks = args.hidden_size, args.hidden_size, args.hidden_size, args.num_layer
        super(ResMLP, self).__init__()
        self.blocks = nn.Sequential(
            *(ResBlock(hidden_dim) for _ in range(num_blocks))
        )
        self.final = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.blocks(x)
        out = self.final(out)
        return out