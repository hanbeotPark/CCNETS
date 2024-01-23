'''
COPYRIGHT (c) 2022. CCNets. All Rights reserved.
'''
import torch
import random
import numpy as np
import torch.nn.functional as F
from torch import nn
from collections.abc import Iterable


def get_loss(loss):
    return loss.mean().detach().cpu().item()

def get_error(loss):
    return loss.mean().detach().cpu().item()

def get_losses(losses):
    ls = []
    for error in losses:
        ls.append(error.mean().detach().cpu().item())
    return np.array(ls)

def get_errors(errors):
    ls = []
    for error in errors:
        ls.append(error.mean().detach().cpu().item())
    return np.array(ls)


def set_random_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

# Mapping activation functions to their PyTorch counterparts
ACTIVATION_FUNCTIONS = {
    "softmax": nn.Softmax(dim=-1),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "relu": nn.ReLU()
}

def add_activation_to_layers(layers, activation_function):
    """Appends the specified activation function to the given layers."""
    if activation_function.lower() != "none":
        if activation_function in ACTIVATION_FUNCTIONS:
            layers.append(ACTIVATION_FUNCTIONS[activation_function])
        else:
            raise ValueError(f"Unsupported activation function: {activation_function}")

class RepeatTensor(nn.Module):
    def __init__(self, n_mul):
        super(RepeatTensor, self).__init__()
        self.repeats = [1, n_mul]
        
    def forward(self, x):
        return x.repeat(*self.repeats)
    
def create_layer(input_size = None, output_size = None, first_act="none", last_act="none"):
    """Creates a PyTorch layer with optional input and output sizes, and optional activation functions."""
    def dumb(x):
        return x
    layers = []
    add_activation_to_layers(layers, first_act)
    if (input_size is not None) and (output_size is not None):  
        layers.append(nn.Linear(input_size, output_size))
    else: 
        return dumb
    add_activation_to_layers(layers, last_act)
    return nn.Sequential(*layers)

def init_weights(param_object):
    if isinstance(param_object, Iterable):
        for layer in param_object:
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.MultiheadAttention):
                nn.init.xavier_uniform_(layer.in_proj_weight)
                nn.init.zeros_(layer.in_proj_bias)
            else:
                # Handle other layer types if needed
                pass
    else:
        if isinstance(param_object, nn.Linear) or isinstance(param_object, nn.Conv2d) or isinstance(param_object, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(param_object.weight)
            nn.init.zeros_(param_object.bias)
        elif isinstance(param_object, nn.MultiheadAttention):
            nn.init.xavier_uniform_(param_object.in_proj_weight)
            nn.init.zeros_(param_object.in_proj_bias)
        else:
            # Handle other layer types if needed
            pass

def select_seq_batch(seq_source_batch, seq_target_batch, seq_len):
    # Get the maximum sequence length
    max_seq_length = seq_source_batch.size(1)
    # Randomly select the starting indices for the subsequence
    start_indices = [random.randint(0, max_seq_length - seq_len) for _ in range(seq_source_batch.size(0))]

    # Select the random subsequences using start_indices
    source_batch = torch.stack([seq_source_batch[j, start_indices[j]:start_indices[j] + seq_len, :] for j in range(seq_source_batch.size(0))])
    target_batch = torch.stack([seq_target_batch[j, start_indices[j]:start_indices[j] + seq_len]    for j in range(seq_target_batch.size(0))])
    return source_batch, target_batch

def convert_to_one_hot_vector(target_batch, label_size):
    return F.one_hot(target_batch.type(torch.int64), num_classes = label_size)

def multiply_list_elements(lst):
    result = 1
    for element in lst:
        result *= element
    return result

