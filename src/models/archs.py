
# IMPORTS 
import torch, time, math
import torch.nn as nn
from random import random
from torch import nn, einsum, norm
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTM, GRU
from torch.nn.modules.normalization import LayerNorm
from torch_geometric.nn import GATConv
import torch.nn.functional as Func
from torch.nn.functional import pad as F_pad



#Create Serializable module: Extends nn.Module. 
# It adds functionality to save, load, and create modules based on a registry of subclasses

class SerializableModule(nn.Module):

    subclasses = {} #a dictionary that saves model names as keys and model classes as values
    
    def __init__(self):
        super().__init__()
        
    # a decorator (@) recieves a class or function as argument and retunrs 
    # a new class or function with modified behaviour

    @classmethod
    def register_model(cls, model_name):
        #gets model class returns class that registers the model in the subclass dinctionary
        def decorator(subclass):
            cls.subclasses[model_name] = subclass 
            return subclass

        return decorator
    
    @classmethod
    def create(cls, arc, **kwargs):
        # gets model class and return the value of the dictionary (i.e. the model class) w/ its args
        if arc not in cls.subclasses:
            raise ValueError('Bad model name {}'.format(arc))

        return cls.subclasses[arc](**kwargs)

    #Saves the module's state dictionary (parameters) to a file.
    def save(self, filename):
        torch.save(self.state_dict(), filename +'.pt')

    #save the architecture and parameters
    def save_entire_model(self, filename):
        torch.save(self, filename +'_entire.pt')

    #load the state dictionary from a file into an instance
    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))


def stat_std_hf(name, h):
    # h: [B,T,F,H] or [B*F,H,T]
    x = h.detach().float().cpu().reshape(-1)
    dh = h[:,1:] - h[:,:-1]
    e = dh.pow(2).mean().item()
    print(f"[{name}] std={x.std().item():.4f} | HF={e:.4f}")


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode='norm'):
        if torch.rand(1) > 0.995 and self.affine==True and self.training: 

            print(f"gamma and beta mean and std, {self.gamma.mean().item():.3f}, {self.gamma.std().item():.3f} and {self.beta.mean().item():.3f}, {self.beta.std().item():.3f}")
        if mode == 'norm':
            self._get_statistics(x)
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)

    def _get_statistics(self, x):
        #self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.mean = torch.median(x, dim=1, keepdim=True).values.detach()
        self.std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        # x: [B, T, F]
        x = (x - self.mean) / self.std
        if self.affine:
            # Reshape para broadcasting automático con [B, T, F]
            x = x * self.gamma.view(1, 1, -1) + self.beta.view(1, 1, -1)
        return x

    def _denormalize(self, x):
        # x: [B, T, F_sub] donde F_sub puede ser target_dim
        mean = self.mean
        std = self.std
        if self.affine:
            gamma = self.gamma
            beta = self.beta
            x = (x - beta.view(1, 1, -1)) / gamma.view(1, 1, -1)
        x = x * std + mean
        return x

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # Only normalizes by RMS, no mean subtraction → preserves offset/trend
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.scale
    
class NestedSTBlockConf(nn.Module):
    def __init__(self, hidden_dim, n_heads=4,
                 kernel_size=3, dilations=[1, 2, 4], dropout=0.1):
        super().__init__()

        self.dilations = dilations
        self.kernel_size = kernel_size
        self.drop = nn.Dropout(dropout)

        # attention over FEATURES
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim,
                      kernel_size, dilation=d,
                      padding=0, bias=False)
            for d in dilations
        ])

        self.norms = nn.ModuleList([RMSNorm(hidden_dim) for _ in dilations])

    def forward(self, h, block_attn=False, block_tcn=False):

        # h: [B, T, F, H]
        B, T, F, H = h.shape

        printin = self.training and torch.rand(1) > 0.995
        if printin:
            stat_std_hf(f"INPUT BACKBONE", h)

        for i, (conv, d, norm) in enumerate(zip(self.convs, self.dilations, self.norms)):

            x = h

            if not block_attn:
                attn_in = x.reshape(B * T, F, H)
                attn_out, _ = self.attn(attn_in, attn_in, attn_in)
                attn_out = attn_out.view(B, T, F, H)
                if printin:
                    stat_std_hf(f"[D{d}] ATTENTION_out", attn_out)
            else:
                attn_out = torch.zeros_like(x)

            x = x + attn_out # B, T, F, H
            if printin:
                stat_std_hf(f"[D{d}] AFTER_attn_add", x)

            if not block_tcn:
                conv_in = x.permute(0, 2, 3, 1)   # B F H T
                conv_in = conv_in.reshape(B * F, H, T)
                pad = (self.kernel_size - 1) * d
                tcn = conv(Func.pad(conv_in, (pad, 0)))
                tcn=tcn.view(B, F, H, T).permute(0, 3, 1, 2)
                tcn = self.drop(Func.elu(tcn))
                if printin:
                    stat_std_hf(f"[D{d}] CONV_out", tcn)
            else:
                tcn = torch.zeros_like(x)
            x = x + tcn
            if printin:
                stat_std_hf(f"[D{d}] AFTER_conv_add", x)

            h = norm(x)
            if printin:
                stat_std_hf(f"[D{d}] AFTER_norm", h)
                print("\n")

        return h

@SerializableModule.register_model('NestedAD')
class NestedAD(SerializableModule):
    def __init__(self, data_dim, activation="relu", norm="layer", hidden_dim=32, kernel_size=5, dilations=[1, 2, 4, 8], attn_heads=4, dropout=0.1, k=6, bottleneck_dim=8):
        super(NestedAD, self).__init__()
        self.data_dim = data_dim
        self.target_dim = data_dim - 6
        self.hidden_dim = hidden_dim

        self.revin = RevIN(num_features=data_dim, affine=False) 
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=1, padding=0, bias=False),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        
        # Nested ST backbone
        self.backbone = NestedSTBlockConf(hidden_dim, kernel_size=kernel_size,
                                      dilations=dilations, dropout=dropout)

        bn=16
        self.head_rec = nn.Sequential(
            nn.Conv1d(hidden_dim, bn, kernel_size=1, bias=False), # pointwise compress
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Conv1d(bn, 1, kernel_size=1),
        )
 
        self.head_fc = nn.Sequential(
            nn.Linear(hidden_dim,bn, bias=False),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(bn, 1),
        )
        

    def forward(self, x, block_tcn=False, block_attn=False, revin=True):
        B,T,F_ext = x.shape
        H = self.hidden_dim
        F = self.target_dim
        
        printin=False
        if self.training and torch.rand(1) > 0.99: 
            printin=True

        # RevIN normalization
        if revin:
            x_main = x[:, :, :F]
            x_c  = x[:, :, F:]
            x_main = self.revin(x_main, mode='norm')
            x = torch.cat([x_main, x_c], dim=-1)
            
        # Input projection
        x_in = x.permute(0,2,1).reshape(B*F_ext,1,T)
        h = self.input_proj(x_in)  # [B*F,H,T]
        
        if printin:
            stat_std_hf("input_proj", h)
               
        h = h.permute(0,2,1)               # [B*F, T, H]
        h = h.view(B, F_ext, T, H)
        h = h.permute(0,2,1,3)             # [B, T, F, H]
        h_st = self.backbone(h, block_attn=block_attn, block_tcn=block_tcn) # [B,T,F,H]
        
        if printin: 
            stat_std_hf("backbone_out", h_st)

        # Bottleneck
        h_rec_in = h_st[:, :, :F, :].permute(0,2,3,1).reshape(B*F,H,T)
        h_fc_in = h_st[:, -1, :F, :].reshape(B*F,H)

        # Reconstruction
        x_hat = self.head_rec(h_rec_in).view(B,F,T).permute(0,2,1)
        if printin:
            stat_std_hf("reconstruction", x_hat)

        # Forecasting
        x_next = self.head_fc(h_fc_in).view(B,F)
        if printin:
            stat_std_hf("forecasting", x_next)
            print("\n")

        # RevIN denorm
        if revin:
            x_hat = self.revin(x_hat, mode='denorm')
            x_next = self.revin(x_next.unsqueeze(1), mode='denorm').squeeze(1)

        return x_hat, x_next
