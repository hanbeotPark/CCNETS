'''
    Author:
        PARK, JunHo, junho.zerotoone@gmail.com
    Reference:
        [1] Ashish Vaswani, et al. Attention Is All You Need
        (https://arxiv.org/abs/1706.03762)

    
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import math
import numpy as np

def positional_encoding(max_seq_size: int, d_model: int) -> torch.Tensor:
    """Creates positional encodings to be added to input tensors."""
    pos_enc = torch.zeros(max_seq_size, d_model)
    pos = torch.arange(max_seq_size, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
    pos_enc[:, 0::2] = torch.sin(pos * div_term)
    pos_enc[:, 1::2] = torch.cos(pos * div_term)
    return pos_enc

def create_look_ahead_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Creates a look-ahead mask for sequence data."""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask

class TFEncoder(nn.Module):
    def __init__(self, args, num_heads: int = 8):
        super(TFEncoder, self).__init__()   
        hidden_size, seq_len, dropout_rate = args.hidden_size, args.seq_len, args.dropout_rate
        self.device = args.device
        self.pos_enc = positional_encoding(max_seq_size = seq_len, d_model = hidden_size).to(self.device)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4, dropout=dropout_rate)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=args.num_layer)
        self.mask = create_look_ahead_mask(seq_len, self.device)
    def forward(self, x):
        x_embed = x + self.pos_enc[:x.size(1), :]
        x_embed = x_embed.permute(1, 0, 2)  # Permute to (seq_len, batch_size, hidden_size) for transformer input
            
        y = self.encoder(x_embed)
        y = y.permute(1, 0, 2)
        return y

class TFDecoder(nn.Module):
    def __init__(self, args, num_heads: int = 8):
        super(TFDecoder, self).__init__()   
        hidden_size, seq_len, dropout_rate = args.hidden_size, args.seq_len, args.dropout_rate
        self.device = args.device
        self.pos_enc1 = positional_encoding(max_seq_size = seq_len, d_model = hidden_size).to(self.device)
        self.pos_enc2 = positional_encoding(max_seq_size = seq_len, d_model = hidden_size).to(self.device)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=hidden_size * 4, dropout=dropout_rate)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=args.num_layer)
        self.tgt_mask = create_look_ahead_mask(seq_len, self.device)
        self.memory_mask = create_look_ahead_mask(seq_len, self.device)

    def forward(self, x, ctx):

        x_embed = x + self.pos_enc1[:x.size(1), :]
        ctx_embed = ctx + self.pos_enc2[:ctx.size(1), :]
        x_embed = x_embed.permute(1, 0, 2)  # Permute to (seq_len, batch_size, hidden_size) for transformer input
        ctx_embed = ctx_embed.permute(1, 0, 2)  # Permute to (seq_len, batch_size, hidden_size) for transformer input

        tgt_mask = self.tgt_mask[:x.size(1), :x.size(1)]
        memory_mask = self.memory_mask[:ctx.size(1), :ctx.size(1)]

        # Decode the action sequence
        y = self.decoder(x_embed, ctx_embed, tgt_mask = tgt_mask, memory_mask = memory_mask)
        y = y.permute(1, 0, 2)  # Permute back to (batch_size, seq_len, hidden_size) for fc layer
        return y
    