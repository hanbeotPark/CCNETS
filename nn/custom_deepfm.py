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
import torch.nn as nn
import torch.nn.functional as F

class DeepFM(nn.Module):
    def __init__(self, args):
        super(DeepFM, self).__init__()
        hidden_size = args.hidden_size  

        state_dim, num_layer = hidden_size, args.num_layer
        output_dim = hidden_size
        self.state_dim = state_dim
        self.num_layer = num_layer
        
        # Define hidden layers for 2nd order terms
        self.dnn = nn.ModuleList()
        for i in range(num_layer):
            self.dnn.append(nn.Linear(hidden_size, hidden_size))
        # Define output layer
        self.output_layer = nn.Linear(hidden_size, output_dim)
        
        
    def forward(self, x):
        # Expand the input tensor along the feature dimension
        # Interaction terms
        
        interactions = torch.matmul(x.unsqueeze(2), x.unsqueeze(1))
        interactions = F.relu(interactions.sum(2))
        # Hidden layers for 2nd order terms
        hidden_output = x.view(-1, self.state_dim)
        for i in range(self.num_layer):
            hidden_output = F.relu(self.dnn[i](hidden_output))
        
        # Output layer
        output = self.output_layer((hidden_output + interactions)/2.)
        
        # Add linear terms and hidden layer outputs element-wise
        return output