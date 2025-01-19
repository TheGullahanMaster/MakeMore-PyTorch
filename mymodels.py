"""
you give this script some words (one per line) and it will generate more things like it.
uses super state of the art Transformer AI tech
this code is intended to be super hackable. tune it to your needs.
Adapted thoroughly from the original script by Andrej Karpathy

Changes from minGPT:
- I removed the from_pretrained function where we init with GPT2 weights
- I removed dropout layers because the models we train here are small,
  it's not necessary to understand at this stage and at this scale.
- I removed weight decay and all of the complexity around what parameters are
  and are not weight decayed. I don't believe this should make a massive
  difference at the scale that we operate on here.

Changes from original MakeMore script:
- Changed the simple RNN into a SiLU RNN
- Added the RNN variants: LSTM, GRU, Light Recurrent Unit (Original from Lucidrains), IndRNN, ReZero Recurrent Unit from (Gates are not what you need, arXiv:2108.00527 [cs.LG]), IndyGRU (Ported from Tensorflow addons, 1.15.x)
- Made all RNNs residual, adding layernorms, variational dropout, zoneout and weight trying
- Added a few transformer variants (GPT-2 with Parabolic Cone activations, RWKV-v5 from a pure pytorch implementation, MinGRU from Lucidrains, HyperMixer (wip) using ChatGPT and Gemini alongside original paper)
- Added ADALIN (fully linear neural network)
- Added a simple sigmoidal MLP
- Changed MLP LM into a residual, SiLU activated MLP ResNet
- Added trainable piecewise linear functions using "torchpwl" library, and implemented in two MLP variants
- Added "attention-less GPT-2 MLP block" as an MLP variant, now with trainable PWLs
- Added the Temporal ConvNet using ChatGPT
- Added Optuna for hyperparameter tuning, uses TPE sampler and the HyperBand pruner
- Added temperature setting
- Added a non-line delimited text file option
- Changed sampling to include a simple "border", created because of the new non-line delimited option
- Added many new optimizers, many in lamb.py
|- Stochastic Gradient Descent, alongside it's momentum counterpart
|- multiple Adam variants
|- Adagrad and AdaDelta
|- NSGDA (wip)
|- Hypergradient versions of AdamW and Lamb
|- GrokFast variants of AdamW and Lamb
|- RMSProp
- A "linegen.sh" script for quick and (hopefully) easy parameter tuning
|- Model Selection
|- Optimizer Selection
|- Seq Len selection (for non-line delimited files)
|- Handles training, resuming training and sampling
- Added a function to store activations (activated by setting num_samples to 1 for now) for each token as it is generated, and an HTML page to visualize them
- Added argmax sampling (Sample count in linegen.sh set to 0), usefull for model's trained on higly deterministic data (math for instance)
- Parabolic Cone Activation function, used in Transformer with Parabolic Cone's
- Added multiple temperature sampling (1.0, 0.8, 0.5, 0.2), inspired by rnntextgen
- Added multipliers to learning rates of certain optimizers (mostly SGD variants), as some require larger learning rates. This system will be replaced by per-optimizer learning rate ranges at some point in the future
- Added LSTM (hiddens, cells) handling into the global RNN class, and into the sampling function class
- Removed restrictions on generated sample size
- Used preexisting sliding window implementation for extended sample generation of non RNN model types
- Added a fully sequential (one to one with hidden states) sampling for RNN variants and the minGRU
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList
import torch
from typing import Tuple
from pau import PAU
import os
import sys
import time
import math
import argparse
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from lamb import *
#from torch.utils.tensorboard import SummaryWriter
import torch
from torch.nn import Parameter, ParameterList
import torch.nn as nn
import torch.nn.functional as F
import math
import math, warnings
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

import math
import functools
from itertools import zip_longest

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange

from beartype import beartype
from beartype.typing import Tuple, Union

from tqdm import tqdm
import torchpwl
from pau import PAU
from torchpwl import PWL
from hypergrad import SGDHD, AdamHD
from adamp import AdamP
from adabelief_pytorch import AdaBelief
### Imports
from collections import deque
from typing import Dict, Optional, Literal
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import sys
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna
from tqdm import tqdm
from grok import GrokFastAdamW
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from optuna.exceptions import TrialPruned

import torch
import torch.nn as nn
import torch
import torch.nn as nn
import json
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path

import torch
import torch.nn as nn
import json
from pathlib import Path



class ParabolicConeActivation(nn.Module):
    def __init__(self, num_features, is_conv=False):
        super(ParabolicConeActivation, self).__init__()
        shape = (1, num_features, 1, 1) if is_conv else (num_features,)
        self.beta = nn.Parameter(torch.full(shape, 2.0))
        self.alpha = nn.Parameter(torch.full(shape, 1.0))
        self.gamma = nn.Parameter(torch.full(shape, 1.0))
        self.sigma = nn.Parameter(torch.full(shape, 1.0))

    def forward(self, x):
        #x = x * self.sigma
        return (x * (2 - x))


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import random
import copy
from tqdm import tqdm

# Define all possible activation functions
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ParabolicConeActivation(nn.Module):
    def forward(self, x):
        return x * (2 - x)

class combHsine(nn.Module):
    def __init__(self, alpha=0.03):
        super().__init__()
        self.alpha = alpha

    def forward(self, input):
        return torch.sinh(self.alpha * input) + torch.asinh(self.alpha * input)

class ConeActivation(nn.Module):
    def forward(self, x):
        return 1 - torch.abs(x - 1)

class GLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GLU, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)
        self.act = torch.sigmoid

    def forward(self, x):
        return self.linear(x) * self.act(self.gate(x))


class LearnableWeightedActivation(nn.Module):
    def __init__(self):
        super(LearnableWeightedActivation, self).__init__()
        self.activations = nn.ModuleList([
            nn.Tanh(),
            nn.ReLU(),
            nn.Sigmoid(),
            nn.PReLU(),
            nn.ELU(),
            nn.SELU(),
            nn.LeakyReLU(),
            nn.SiLU(),
            nn.GELU(),
            Swish(),
            ParabolicConeActivation(),
            combHsine(),
            ConeActivation(),
            # Add more activation functions here
        ])
        self.weights = nn.Parameter(torch.ones(len(self.activations)))  # Initialize weights
        # Initialize weights with ones to start equally
        nn.init.constant_(self.weights, 1.0)

    def forward(self, x):
        # Apply each activation
        activations_out = [act(x) for act in self.activations]
        # Stack activations along a new dimension
        stacked = torch.stack(activations_out, dim=1)  # Shape: [batch, num_acts, ...]
        # Apply softmax to weights
        weight_softmax = F.softmax(self.weights, dim=0)  # Shape: [num_acts]
        # Reshape weights for broadcasting
        weight_softmax = weight_softmax.view(1, -1, *([1] * (stacked.dim() - 2)))
        # Weighted sum of activations
        output = torch.sum(stacked * weight_softmax, dim=1)
        return output

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * (self.gamma + 1)

class LAuReL(nn.Module):
    def __init__(
        self,
        input_dim,
        version='RW',
        rank=None,
        use_previous_activations=False,
        num_layers=None
    ):
        """
        Args:
            input_dim (int): Dimension of the input features.
            version (str): Version of LAuReL to use ('RW', 'LR', or 'PA').
            rank (int, optional): Rank for the low-rank approximation in 'LR' and 'PA' versions.
            use_previous_activations (bool): Whether to use previous activations ('PA' version).
            num_layers (int, optional): Number of layers (required if use_previous_activations=True).
        """
        super(LAuReL, self).__init__()
        self.version = version
        self.use_previous_activations = use_previous_activations
        self.input_dim = input_dim
        
        if self.version == 'RW' or self.version == 'RW+LR':
            # Learnable scalar weights with softmax normalization
            self.alpha = nn.Parameter(torch.randn(1))
            self.beta = nn.Parameter(torch.randn(1))
        
        if self.version == 'LR' or self.version == 'RW+LR':
            assert rank is not None, "Rank must be specified for low-rank version."
            self.rank = rank
            # Low-rank matrices A and B for the transformation W = A @ B + I
            self.A = nn.Parameter(torch.randn(input_dim, rank))
            self.B = nn.Parameter(torch.randn(rank, input_dim))
        
        if self.use_previous_activations:
            assert num_layers is not None, "Number of layers must be specified when using previous activations."
            self.num_layers = num_layers
            # Learnable scalar weights gamma for previous activations
            self.gamma = nn.Parameter(torch.randn(num_layers))
            # Optional low-rank transformation h(x)
            if rank is not None:
                self.h_A = nn.Parameter(torch.randn(input_dim, rank))
                self.h_B = nn.Parameter(torch.randn(rank, input_dim))
            else:
                self.h_W = nn.Parameter(torch.randn(input_dim, input_dim))

    def forward(self, x, f_x, previous_activations=None):
        """
        Args:
            x (Tensor): Input tensor x_i.
            f_x (Tensor): Output of the non-linear function f(x_i).
            previous_activations (list of Tensors, optional): List of previous activations x_j.
        Returns:
            Tensor: The output tensor x_{i+1}.
        """
        if self.version == 'RW':
            # Normalize alpha and beta using softmax
            weights = F.softmax(torch.cat([self.alpha, self.beta]), dim=0)
            alpha = weights[0]
            beta = weights[1]
            residual = alpha * f_x + beta * x
            return residual

        elif self.version == 'LR':
            W = self.A @ self.B + torch.eye(self.input_dim, device=x.device)
            residual = f_x + W @ x
            return residual

        elif self.version == 'RW+LR':
            # Normalize alpha and beta using softmax
            weights = F.softmax(torch.cat([self.alpha, self.beta]), dim=0)
            alpha = weights[0]
            beta = weights[1]
            W = self.A @ self.B + torch.eye(self.input_dim, device=x.device)
            residual = alpha * f_x + beta * (W @ x)
            return residual

        elif self.version == 'PA':
            assert previous_activations is not None, "Previous activations must be provided for PA version."
            if hasattr(self, 'h_A') and hasattr(self, 'h_B'):
                h = lambda x_j: self.h_A @ (self.h_B @ x_j)
            elif hasattr(self, 'h_W'):
                h = lambda x_j: self.h_W @ x_j
            else:
                h = lambda x_j: x_j  # Identity if no h is defined

            # Compute the weighted sum of previous activations
            gamma_weights = F.softmax(self.gamma[:len(previous_activations)], dim=0)
            g_x = sum(gamma_weights[j] * h(previous_activations[j]) for j in range(len(previous_activations)))

            residual = f_x + g_x
            return residual

        else:
            raise ValueError("Invalid version specified for LAuReL.")



### Grokfast
def gradfilter_ema(
    m: nn.Module,
    grads: Optional[Dict[str, torch.Tensor]] = None,
    alpha: float = 0.99,
    lamb: float = 5.0,
) -> Dict[str, torch.Tensor]:
    if grads is None:
        grads = {n: p.grad.data.detach() for n, p in m.named_parameters() if p.requires_grad}

    for n, p in m.named_parameters():
        if p.requires_grad:
            grads[n] = grads[n] * alpha + p.grad.data.detach() * (1 - alpha)
            p.grad.data = p.grad.data + grads[n] * lamb

    return grads


### Grokfast-MA
def gradfilter_ma(
    m: nn.Module,
    grads: Optional[Dict[str, deque]] = None,
    window_size: int = 128,
    lamb: float = 5.0,
    filter_type: Literal['mean', 'sum'] = 'mean',
    warmup: bool = True,
    trigger: bool = False,
) -> Dict[str, deque]:
    if grads is None:
        grads = {n: deque(maxlen=window_size) for n, p in m.named_parameters() if p.requires_grad}

    for n, p in m.named_parameters():
        if p.requires_grad:
            grads[n].append(p.grad.data.detach())

            if not warmup or len(grads[n]) == window_size and not trigger:
                if filter_type == "mean":
                    avg = sum(grads[n]) / len(grads[n])
                elif filter_type == "sum":
                    avg = sum(grads[n])
                else:
                    raise ValueError(f"Unrecognized filter_type {filter_type}")
                p.grad.data = p.grad.data + avg * lamb

    return grads

class MVPOG(nn.Module):
    def __init__(self, hidden_size, num_breakpoints):
        super(MVPOG, self).__init__()
        self.hidden_size = hidden_size
        self.pwl = PWL(num_channels=hidden_size, num_breakpoints=num_breakpoints)

    def forward(self, x):
        # Reshape to (batch_size, num_channels, 1, 1)
        x = x.view(x.size(0), self.hidden_size, 1, 1)
        
        # Apply PWL
        x = self.pwl(x)
        
        # Reshape back to (batch_size, hidden_size)
        x = x.view(x.size(0), self.hidden_size)
        
        return x

import torch.nn as nn

class MVP(nn.Module):
    def __init__(self, hidden_size, num_breakpoints):
        super(MVP, self).__init__()
        self.hidden_size = hidden_size
        self.pwl = PWL(num_channels=hidden_size, num_breakpoints=num_breakpoints)
        self.layernorm = nn.LayerNorm(hidden_size)  # Add LayerNorm initialization

    def forward(self, x):
        batch_size, block_size, hidden_size = x.size()

        # Apply LayerNorm to the input
        x = self.layernorm(x)
        
        # Reshape to (batch_size * block_size, hidden_size, 1, 1)
        x = x.view(batch_size * block_size, hidden_size, 1, 1)
        
        # Apply PWL
        x = self.pwl(x)
        
        # Reshape back to (batch_size, block_size, hidden_size)
        x = x.view(batch_size, block_size, hidden_size)
        
        return x


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def remainder_to_mult(num, mult):
    return (mult - num % mult) % mult

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def reduce_mult(nums):
    return functools.reduce(lambda x, y: x * y, nums, 1)

# tensor helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# token shift, from Peng et al of RWKV

def token_shift(t):
    t, t_shift = t.chunk(2, dim = -1)
    t_shift = F.pad(t_shift, (0, 0, 1, -1))
    return torch.cat((t, t_shift), dim = -1)
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ComplexLinear, self).__init__()
        self.real_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.imag_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.real_bias = nn.Parameter(torch.Tensor(out_features))
            self.imag_bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('real_bias', None)
            self.register_parameter('imag_bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.real_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.imag_weight, a=math.sqrt(5))
        if self.real_bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.real_weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.real_bias, -bound, bound)
            nn.init.uniform_(self.imag_bias, -bound, bound)

    def forward(self, input):
        real = F.linear(input.real, self.real_weight, self.real_bias) - F.linear(input.imag, self.imag_weight, self.imag_bias)
        imag = F.linear(input.real, self.imag_weight, self.imag_bias) + F.linear(input.imag, self.real_weight, self.real_bias)
        return torch.complex(real, imag)

# Custom complex activation function (SiLU)
class ComplexSiLU(nn.Module):
    def __init__(self, hidd):
        super(ComplexSiLU, self).__init__()
        self.act_real = nn.SiLU()
        self.act_imag = nn.SiLU()
    
    def forward(self, input):
        return torch.complex(self.act_real(input.real), self.act_imag(input.imag))

# Custom complex layer normalization
class ComplexLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(ComplexLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape, dtype=torch.complex64))
        self.bias = nn.Parameter(torch.zeros(normalized_shape, dtype=torch.complex64))
        self.eps = eps

    def forward(self, input):
        mean = input.mean(-1, keepdim=True)
        input = input - mean
        var = (input.real ** 2 + input.imag ** 2).mean(-1, keepdim=True)
        std = torch.sqrt(var + self.eps)
        input = input / std
        return self.weight * input + self.bias
# -----------------------------------------------------------------------------

@dataclass
class ModelConfig:
    block_size: int = None # length of the input sequences of integers
    vocab_size: int = None # the input integers are in range [0 .. vocab_size -1]
    # parameters below control the sizes of each model slightly differently
    n_layer: int = 4
    n_embd: int = 64
    n_embd2: int = 64
    n_head: int = 4

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    @property
    def device(self):
        return next(self.buffers()).device

    def forward(self, seq_len):
        t = torch.arange(seq_len, device = self.device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)
        return freqs

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(pos, t):
    return t * pos.cos() + rotate_half(t) * pos.sin()

# -----------------------------------------------------------------------------
# Transformer Language Model (*exactly* as used in GPT-2)

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y
class GPTOGBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act1     = nn.SiLU(),
        ))
        m = self.mlp
        #self.pau1 = MVP(config.n_embd, 16)
        #self.pau2 = MVP(config.n_embd, 16)
        self.mlpf = lambda x: m.c_proj(m.act1(m.c_fc(x))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x
class GPTBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act1     = ParabolicConeActivation(),
            actin     = ParabolicConeActivation(),
            actout     = ParabolicConeActivation(),
        ))
        m = self.mlp
        #self.pau1 = MVP(config.n_embd, 50)
        #self.pau2 = MVP(config.n_embd, 50)
        self.mlpf = lambda x: m.c_proj(m.act1(m.c_fc(x))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class RWKV_TimeMix_x051a(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.head_size = config.n_embd // config.n_head
        self.n_head = config.n_head

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (config.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                ddd[0, 0, i] = i / config.n_embd

            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            decay_speed = torch.ones(self.n_head)
            for h in range(self.n_head):
                decay_speed[h] = -6 + 5 * (h / (self.n_head - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.unsqueeze(-1))

            tmp = torch.zeros(self.n_head)
            for h in range(self.n_head):
                tmp[h] = ratio_0_to_1 * (1 - (h / (self.n_head - 1)))
            self.time_faaaa = nn.Parameter(tmp.unsqueeze(-1))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.gate = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.output = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, config.n_embd, eps=(1e-5)*64)

        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        H, N = self.n_head, self.head_size
        #
        # we divide a block into chunks to speed up computation & save vram.
        # you can try to find the optimal chunk_len for your GPU.
        # avoid going below 128 if you are using bf16 (otherwise time_decay might be less accurate).
        #
        if T % 256 == 0: Q = 256
        elif T % 128 == 0: Q = 128
        else:
            Q = T
            warnings.warn(f'\n{"#"*80}\n\n{" "*38}Note\nThe GPT-mode forward() should only be called when we are training models.\nNow we are using it for inference for simplicity, which works, but will be very inefficient.\n\n{"#"*80}\n')
        assert T % Q == 0

        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xv = x + xx * self.time_maa_v
        xr = x + xx * self.time_maa_r
        xg = x + xx * self.time_maa_g
        r = self.receptance(xr).view(B, T, H, N).transpose(1, 2) # receptance
        k = self.key(xk).view(B, T, H, N).permute(0, 2, 3, 1) # key
        v = self.value(xv).view(B, T, H, N).transpose(1, 2) # value
        g = F.silu(self.gate(xg)) # extra gate

        w = torch.exp(-torch.exp(self.time_decay.float())) # time_decay
        u = self.time_faaaa.float() # time_first

        ws = w.pow(Q).view(1, H, 1, 1)

        ind = torch.arange(Q-1, -1, -1, device=r.device).unsqueeze(0).repeat(H, 1)
        w = w.repeat(1, Q).pow(ind)

        wk = w.view(1, H, 1, Q)
        wb = wk.transpose(-2, -1).flip(2)

        w = torch.cat([w[:, 1:], u], dim=1)
        w = F.pad(w, (0, Q))
        w = torch.tile(w, [Q])
        w = w[:, :-Q].view(-1, Q, 2*Q - 1)
        w = w[:, :, Q-1:].view(1, H, Q, Q)

        w = w.to(dtype=r.dtype) # the decay matrix
        wk = wk.to(dtype=r.dtype)
        wb = wb.to(dtype=r.dtype)
        ws = ws.to(dtype=r.dtype)

        state = torch.zeros(B, H, N, N, device=r.device, dtype=r.dtype) # state
        y = torch.empty(B, H, T, N, device=r.device, dtype=r.dtype) # output

        for i in range(T // Q): # the rwkv-x051a operator
            rr = r[:, :, i*Q:i*Q+Q, :]
            kk = k[:, :, :, i*Q:i*Q+Q]
            vv = v[:, :, i*Q:i*Q+Q, :]
            y[:, :, i*Q:i*Q+Q, :] = ((rr @ kk) * w) @ vv + (rr @ state) * wb
            state = ws * state + (kk * wk) @ vv

        y = y.transpose(1, 2).contiguous().view(B * T, C)
        y = self.ln_x(y).view(B, T, C) * g

        # output projection
        y = self.dropout(self.output(y))
        return y

class RWKV_ChannelMix_x051a(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / config.n_layer)
            ddd = torch.ones(1, 1, config.n_embd)
            for i in range(config.n_embd):
                ddd[0, 0, i] = i / config.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.value = nn.Linear(3 * config.n_embd, config.n_embd, bias=False)
        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        x = self.key(xk)
        x = torch.relu(x) ** 2
        x = self.value(x)
        x = torch.sigmoid(self.receptance(xr)) * x
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config, layer_id):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=False)
        self.tmix = RWKV_TimeMix_x051a(config, layer_id)
        self.ln_2 = LayerNorm(config.n_embd, bias=False)
        self.cmix = RWKV_ChannelMix_x051a(config, layer_id)

    def forward(self, x):
        x = x + self.tmix(self.ln_1(x))
        x = x + self.cmix(self.ln_2(x))
        return x

class Transformer(nn.Module):
    """ Transformer Language Model, exactly as seen in GPT-2 """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        #self.mvpin = MVP(config.n_embd, 50)
        self.mvpout = MVP(config.vocab_size, 50)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([GPTBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.mvpout(self.lm_head(x))

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

class OGTransformer(nn.Module):
    """ Transformer Language Model, exactly as seen in GPT-2 """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([GPTOGBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss
class RWKV5(nn.Module):
    """ Transformer Language Model, exactly as seen in GPT-2 """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config, _) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        #pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = tok_emb# + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

# -----------------------------------------------------------------------------
# Bag of Words (BoW) language model

class CausalBoW(nn.Module):
    """
    Causal bag of words. Averages the preceding elements and looks suspiciously like
    a CausalAttention module you'd find in a transformer, for no apparent reason at all ;)
    """
    def __init__(self, config):
        super().__init__()

        # used to mask out vectors and preserve autoregressive property
        self.block_size = config.block_size
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, n_embd

        # do the weighted average of all preceeding token features
        att = torch.zeros((B, T, T), device=x.device)
        att = att.masked_fill(self.bias[:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ x # (B, T, T) x (B, T, C) -> (B, T, C)

        return y

class BoWBlock(nn.Module):
    """ collects BoW features and adds an MLP """

    def __init__(self, config):
        super().__init__()

        # Causal BoW module
        self.cbow = CausalBoW(config)
        # MLP assembler
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, config.n_embd2),
            c_proj  = nn.Linear(config.n_embd2, config.n_embd),
        ))
        m = self.mlp
        self.act = nn.SiLU()
        self.mlpf = lambda x: m.c_proj(self.act(m.c_fc(x))) # MLP forward

    def forward(self, x):
        x = x + self.cbow(x)
        x = x + self.mlpf(x)
        return x

class BoW(nn.Module):
    """
    takes the previous block_size tokens, encodes them with a lookup table,
    also encodes their positions with lookup table, then averages all of those
    embeddings up and uses that to predict the next token.
    """

    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        # token embedding
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # position embedding
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        # context block
        self.context_block1 = BoWBlock(config)
        self.context_block2 = BoWBlock(config)
        self.context_block3 = BoWBlock(config)
        self.context_block4 = BoWBlock(config)
        # language model head decoder layer
        self.lm_head = nn.Linear(config.n_embd, self.vocab_size)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):

        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the token and position embedding layers
        tok_emb = self.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos) # position embeddings of shape (1, t, n_embd)
        # add and run through the decoder MLP
        x = tok_emb + pos_emb
        # run the bag of words context module
        x = self.context_block1(x)
        x = self.context_block2(x)
        x = self.context_block3(x)
        x = self.context_block4(x)
        # decode to next token probability
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

# -----------------------------------------------------------------------------
"""
Recurrent Neural Net language model: either a vanilla RNN recurrence or a GRU.
Did not implement an LSTM because its API is a bit more annoying as it has
both a hidden state and a cell state, but it's very similar to GRU and in
practice works just as well.
"""

class combHsine(nn.Module):
    def __init__(self, alpha=0.03):
        super().__init__()
        self.alpha = alpha

    def forward(self, input):
        return torch.sinh(self.alpha * input) + torch.arcsinh(self.alpha * input)


class RNNCell(nn.Module):
    """
    the job of a 'Cell' is to:
    take input at current time step x_{t} and the hidden state at the
    previous time step h_{t-1} and return the resulting hidden state
    h_{t} at the current timestep
    """
    def __init__(self, config):
        super().__init__()
        self.xh_to_h = nn.Linear(config.n_embd + config.n_embd, config.n_embd)
        self.act = nn.SiLU()

    def forward(self, xt, hprev):
        xh = torch.cat([xt, hprev], dim=1)
        ht = self.act(self.xh_to_h(xh))
        return ht

class IndRNNCell(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.recurrent_weight = nn.Parameter(torch.Tensor(config.n_embd))
        self.input_weight = nn.Linear(config.n_embd, config.n_embd)
        self.act = NewGELU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.recurrent_weight, -0.5, 0.5)
        self.input_weight.reset_parameters()

    def forward(self, xt, hprev):
        ht = self.input_weight(xt) + self.act(hprev * self.recurrent_weight)
        return ht

class GRUCell(nn.Module):
    """
    same job as RNN cell, but a bit more complicated recurrence formula
    that makes the GRU more expressive and easier to optimize.
    """
    def __init__(self, config):
        super().__init__()
        # input, forget, output, gate
        self.xh_to_z = nn.Linear(config.n_embd + config.n_embd, config.n_embd)
        self.xh_to_r = nn.Linear(config.n_embd + config.n_embd, config.n_embd)
        self.xh_to_hbar = nn.Linear(config.n_embd + config.n_embd, config.n_embd)
        self.act = nn.Tanh()

    def forward(self, xt, hprev):
        # first use the reset gate to wipe some channels of the hidden state to zero
        xh = torch.cat([xt, hprev], dim=1)
        r = F.sigmoid(self.xh_to_r(xh))
        hprev_reset = r * hprev
        # calculate the candidate new hidden state hbar
        xhr = torch.cat([xt, hprev_reset], dim=1)
        hbar = self.act(self.xh_to_hbar(xhr))
        # calculate the switch gate that determines if each channel should be updated at all
        z = F.sigmoid(self.xh_to_z(xh))
        # blend the previous hidden state and the new candidate hidden state
        ht = (1 - z) * hprev + z * hbar
        return ht

class IndRNNCell2(nn.Module):
    def __init__(self, config):
        super(IndRNNCell2, self).__init__()
        input_size = config.n_embd
        hidden_size = config.n_embd
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = NewGELU()
        self.u = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.w = nn.Parameter(torch.Tensor(hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.u, a=math.sqrt(5))
        nn.init.ones_(self.w)
        nn.init.zeros_(self.bias)
        self.w.data.uniform_(-1, 1)  # Constrain recurrent weight
        self.w.data = self.w.data.clamp(-1, 1)  # Ensure weights remain constrained

    def forward(self, x, h):
        h_next = self.activation(x @ self.u + self.w * h + self.bias)
        self.w.data = self.w.data.clamp(-1, 1)  # Ensure weights remain constrained during training
        return h_next

def check_bounds(weight, min_abs, max_abs):
    if min_abs:
        abs_kernel = torch.abs(weight).clamp_(min=min_abs)
        weight = torch.mul(torch.sign(weight), abs_kernel)
    if max_abs:
        weight = weight.clamp(max=max_abs, min=-max_abs)
    return weight

import torch
import torch.nn as nn
import torch.nn.functional as F

class IndRNNCell3(nn.Module):
    """
    Independently Recurrent Neural Network (IndRNN) cell.
    Reference: https://arxiv.org/abs/1803.04831

    Args:
        input_size: int, The number of input features.
        hidden_size: int, The number of units in the RNN cell.
        activation: Callable, Activation function to use. Default: `torch.tanh`.
    """

    def __init__(self, config):
        super(IndRNNCell3, self).__init__()
        self.input_size = input_size = config.n_embd#input_size
        self.hidden_size = hidden_size = config.n_embd#hidden_size
        self.activation = activation = nn.PReLU(hidden_size)#ParabolicConeActivation(hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

        # Input-to-hidden weights
        self.kernel_w = nn.Parameter(torch.Tensor(input_size, hidden_size))
        # Recurrent weights (one per hidden unit)
        self.kernel_u = nn.Parameter(torch.Tensor(hidden_size))
        # Bias term
        self.bias = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize input-to-hidden weights
        nn.init.xavier_uniform_(self.kernel_w)
        # Initialize recurrent weights with uniform distribution [-1, 1]
        nn.init.uniform_(self.kernel_u, -1, 1)
        # Initialize bias to zeros
        nn.init.zeros_(self.bias)

    def forward(self, input, hx):
        """
        Args:
            input: Tensor of shape (batch_size, input_size)
            hx: Tensor of shape (batch_size, hidden_size), initial hidden state

        Returns:
            output: Tensor of shape (batch_size, hidden_size), output of the cell
            hx: Tensor of shape (batch_size, hidden_size), new hidden state
        """
        # Compute input-to-hidden contribution
        input_contribution = torch.matmul(input, self.kernel_w)
        # Compute recurrent contribution
        recurrent_contribution = hx * self.kernel_u
        # Combine contributions and add bias
        gate_inputs = input_contribution + recurrent_contribution + self.bias
        # Apply activation function
        output = self.activation(self.norm(gate_inputs))
        # Return output and new hidden state
        return output


import torch
import torch.nn as nn
import torch.nn.functional as F

class IndyGRUCell(nn.Module):
    r"""
    Independently Gated Recurrent Unit (IndyGRU) cell.
    Reference:
    - IndRNN: https://arxiv.org/abs/1803.04831
    - GRU: http://arxiv.org/abs/1406.1078

    This cell is similar to a standard GRU cell but replaces the recurrent weight matrices
    with diagonal matrices (element-wise multiplications with vectors), making each neuron
    only interact with its own past state.

    Args:
        input_size (int): The number of input features.
        hidden_size (int): The number of units in the RNN cell.
        activation (callable, optional): Activation function to use. Default: `torch.tanh`.
        bias (bool, optional): If `False`, then the layer does not use bias weights. Default: `True`.
    """

    def __init__(self, config):
        super(IndyGRUCell, self).__init__()
        self.input_size = input_size = config.n_embd
        self.hidden_size = hidden_size = config.n_embd
        self.activation = activation = nn.SiLU()
        self.norm = nn.LayerNorm(config.n_embd)
        self.bias = bias = True

        # Input-to-hidden weights for gates (W_r and W_z)
        self.weight_ih = nn.Parameter(torch.Tensor(2 * hidden_size, input_size))
        # Hidden-to-hidden weights for gates (u_r and u_z)
        self.weight_hh = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        # Biases for gates
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(2 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)

        # Input-to-hidden weights for candidate (W)
        self.weight_ih_candidate = nn.Parameter(torch.Tensor(hidden_size, input_size))
        # Hidden-to-hidden weights for candidate (u)
        self.weight_hh_candidate = nn.Parameter(torch.Tensor(1, hidden_size))
        # Bias for candidate
        if bias:
            self.bias_ih_candidate = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias_ih_candidate', None)

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_ih_candidate)
        nn.init.uniform_(self.weight_hh, -1, 1)
        nn.init.uniform_(self.weight_hh_candidate, -1, 1)
        # Initialize biases
        if self.bias:
            nn.init.constant_(self.bias_ih, 1.0)
            nn.init.zeros_(self.bias_ih_candidate)

    def forward(self, input, hx):
        """
        Args:
            input (Tensor): Input tensor of shape (batch_size, input_size).
            hx (Tensor): Hidden state tensor of shape (batch_size, hidden_size).

        Returns:
            Tensor: New hidden state tensor of shape (batch_size, hidden_size).
        """
        batch_size = input.size(0)

        # Gates: r_t and z_t

        # Input contribution for gates
        gate_x = F.linear(input, self.weight_ih, self.bias_ih)  # (batch_size, 2 * hidden_size)

        # Tile or concatenate hx to match gate dimensions
        hx_tiled = hx.repeat(1, 2)  # Option A
        # hx_tiled = torch.cat([hx, hx], dim=1)  # Option B

        # Hidden state contribution for gates
        gate_h = hx_tiled * self.weight_hh  # (batch_size, 2 * hidden_size)

        # Total gate input
        gate = gate_x + gate_h  # (batch_size, 2 * hidden_size)

        # Split gates
        r_t, z_t = gate.chunk(2, dim=1)
        r_t = self.activation(r_t)
        z_t = self.activation(z_t)

        # Candidate hidden state
        candidate_x = F.linear(input, self.weight_ih_candidate, self.bias_ih_candidate)  # (batch_size, hidden_size)
        candidate_h = (r_t * hx) * self.weight_hh_candidate  # (batch_size, hidden_size)
        candidate = candidate_x + candidate_h
        c_t = self.activation(candidate)

        # New hidden state
        new_h = z_t * hx + (1 - z_t) * c_t
        return self.norm(new_h)


import torch
import torch.nn as nn
import numpy as np

class RRUCell(nn.Module):
    def __init__(self, config, training=False):
        super(RRUCell, self).__init__()
        num_units = config.n_embd
        output_size=config.n_embd
        relu_layers=1
        middle_layer_size_multiplier=4
        dropout_rate=0.0

        self.num_units = num_units
        self.output_size = output_size
        self.relu_layers = relu_layers
        self.middle_layer_size_multiplier = middle_layer_size_multiplier
        self.dropout_rate = dropout_rate
        self.training = training

        self.J_kernels = nn.ParameterList()
        self.J_biases = nn.ParameterList()

        input_depth = num_units
        total = input_depth + self.num_units
        n_middle_maps = round(self.middle_layer_size_multiplier * total)

        for i in range(self.relu_layers):
            if i == 0:
                j_kernel = nn.Parameter(torch.Tensor(total, n_middle_maps))
                j_bias = nn.Parameter(torch.Tensor(n_middle_maps))
            else:
                j_kernel = nn.Parameter(torch.Tensor(n_middle_maps, n_middle_maps))
                j_bias = nn.Parameter(torch.Tensor(n_middle_maps))

            self.J_kernels.append(j_kernel)
            self.J_biases.append(j_bias)

        self.S_bias_variable = nn.Parameter(torch.Tensor(self.num_units))
        self.W_kernel = nn.Parameter(torch.Tensor(n_middle_maps, self.num_units + self.output_size))
        self.W_bias = nn.Parameter(torch.Tensor(self.num_units + self.output_size))
        self.Z_ReZero = nn.Parameter(torch.Tensor(self.num_units))

        self.init_weights()

    def init_weights(self):
        for param in self.parameters():
            nn.init.normal_(param)

        inv_sigmoid_values = inv_sigmoid(np.random.uniform(0.01, 0.99, size=self.num_units)) / 10.
        with torch.no_grad():
            self.S_bias_variable.copy_(torch.from_numpy(inv_sigmoid_values))

    def forward(self, inputs, state):
        input_and_state = torch.cat([inputs, state], 1)
    
        all_J_kernels = torch.cat([kernel for kernel in self.J_kernels], dim=1) 
        all_J_biases = torch.cat([bias for bias in self.J_biases], dim=0) # Corrected line
    
    
        after_j = torch.matmul(input_and_state, all_J_kernels) + all_J_biases 
        after_j = instance_norm(after_j)  # Assuming instance_norm is optimized
    
        j_start = torch.relu(after_j) 

        if self.training:
            after_dropout = nn.functional.dropout(j_start, self.dropout_rate)
        else:
            after_dropout = j_start

        after_w = torch.matmul(after_dropout, self.W_kernel) + self.W_bias

        output = after_w[:, self.num_units:]
        candidate = after_w[:, 0:self.num_units]

        final_state = state * self.S_bias_variable * 10. + torch.sigmoid(candidate * self.Z_ReZero)

        return (output + inputs), final_state

    def zero_state(self, batch_size, dtype):
        initial = torch.tensor([1] + [0] * (self.num_units - 1), dtype=dtype) * np.sqrt(self.num_units) * 0.25
        return initial.repeat(batch_size, 1)

def instance_norm(cur):
    variance = torch.mean(cur ** 2, dim=-1, keepdim=True)
    cur = cur * torch.rsqrt(variance + 1e-6)
    return cur

def inv_sigmoid(y):
    return np.log(y / (1 - y))

class MogrifierGRUCell(nn.Module):

    def __init__(self, config, mogrify_steps=5):
        super(MogrifierGRUCell, self).__init__()
        self.mogrify_steps = mogrify_steps
        self.lstm = GRUCell(config)
        self.mogrifier_list = nn.ModuleList([nn.Linear(config.n_embd, config.n_embd)])  # start with q
        for i in range(1, mogrify_steps):
            if i % 2 == 0:
                self.mogrifier_list.extend([nn.Linear(config.n_embd, config.n_embd)])  # q
            else:
                self.mogrifier_list.extend([nn.Linear(config.n_embd, config.n_embd)])  # r
   
    def mogrify(self, x, h):
        for i in range(self.mogrify_steps):
            if (i+1) % 2 == 0: 
                h = (2*torch.sigmoid(self.mogrifier_list[i](x))) * h
            else:
                x = (2*torch.sigmoid(self.mogrifier_list[i](h))) * x
        return x, h

    def forward(self, x, states):
        ht = states
        x, ht = self.mogrify(x, ht)
        ht = self.lstm(x, ht)
        return ht

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

class MogrifierLSTMCell(nn.Module):

    def __init__(self, config, mogrify_steps=5):
        super(MogrifierLSTMCell, self).__init__()
        input_size = config.n_embd
        hidden_size = config.n_embd
        attention_heads = 8
        self.mogrify_steps = mogrify_steps
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.self_attention1 = SelfAttention(hidden_size, attention_heads)
        self.self_attention2 = SelfAttention(hidden_size, attention_heads)
        self.mogrifier_list = nn.ModuleList([nn.Linear(hidden_size, input_size)])  # start with q
        for i in range(1, mogrify_steps):
            if i % 2 == 0:
                self.mogrifier_list.extend([nn.Linear(hidden_size, input_size)])  # q
            else:
                self.mogrifier_list.extend([nn.Linear(input_size, hidden_size)])  # r
   
    def mogrify(self, x, h):
        for i in range(self.mogrify_steps):
            if (i+1) % 2 == 0: 
                h = (2*torch.sigmoid(self.mogrifier_list[i](x))) * h
            else:
                x = (2*torch.sigmoid(self.mogrifier_list[i](h))) * x
        return x, h

    def forward(self, x, states):
        hat, cat = states
        ht = self.self_attention1(hat.unsqueeze(0), hat.unsqueeze(0), cat.unsqueeze(0)).squeeze(0) + hat
        ct = self.self_attention2(cat.unsqueeze(0), cat.unsqueeze(0), hat.unsqueeze(0)).squeeze(0) + cat
        #x, ht = self.mogrify(x, ht)
        ht, ct = self.lstm(x, (ht, ct))
        return ht, ct
class LRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, layerindex):
        super(LRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.W_h = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_f = nn.Linear(input_size, hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        #self.b_f = nn.Parameter(torch.zeros(hidden_size))
        self.s_s = nn.Parameter(torch.ones(hidden_size))
        self.layer = layerindex
        self.once = 0
        
        # Replace PReLU with LEAF activation
        self.act1 = nn.SiLU()#torch.tanh#ParabolicConeActivation(hidden_size)#(1)
        self.act2 = nn.SiLU()#F.sigmoid#ParabolicConeActivation(hidden_size)#(1)
        
        # Set self.norm based on layerindex
        #self.norm = 0.5 ** layerindex


    def forward(self, x, h_prev):
        # Compute candidate hidden state
        h_tilde = self.act1(self.norm1(self.W_h(x)))
        
        # Compute forget gate
        f_t = self.act2(self.norm2(self.U_f(h_prev) + self.W_f(x)))
        
        # Update hidden state
        h_t = (self.s_s - f_t) * h_prev + f_t * h_tilde
        
        return self.norm3(h_t)# * self.norm
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, config, cell_type, training, zoneout_prob=0.0, variational_dropout_prob=0.0, ar_alpha=0.0, tar_beta=0.0):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        self.type = cell_type
        self.training = training
        self.config = config
        self.zoneout_prob = zoneout_prob
        self.variational_dropout_prob = variational_dropout_prob
        self.ar_alpha = ar_alpha
        self.tar_beta = tar_beta

        # Token and positional embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(self.block_size, config.n_embd)

        # Initial hidden and cell states
        self.start = nn.Parameter(torch.zeros(self.n_layer, 1, config.n_embd))
        self.start_ct = nn.Parameter(torch.zeros(self.n_layer, 1, config.n_embd))

        # RNN cells
        self.cells = nn.ModuleList()
        for layer in range(self.n_layer):
            if cell_type == 'rnn':
                self.cells.append(IndRNNCell3(config))
            elif cell_type == 'ogrnn':
                self.cells.append(RNNCell(config))
            elif cell_type == 'gru':
                self.cells.append(IndyGRUCell(config))
            elif cell_type == 'moglstm':
                self.cells.append(LRUCell(config.n_embd, config.n_embd, layer))
            elif cell_type == 'rru':
                self.cells.append(RRUCell(config, training))
            elif cell_type == 'lstm':
                self.cells.append(nn.LSTMCell(config.n_embd, config.n_embd))
            elif cell_type == 'oggru':
                self.cells.append(nn.GRUCell(config.n_embd, config.n_embd))
            else:
                raise ValueError(f"Unsupported cell type: {cell_type}")

        # Layer normalization and dropout
        self.layer_norms = nn.ModuleList([nn.LayerNorm(config.n_embd) for _ in range(self.n_layer)])
        self.dropout = nn.Dropout(0.0)

        # Output layer with weight tying
        self.lm_head = nn.Linear(config.n_embd, self.vocab_size)
        self.lm_head.weight = self.wte.weight  # Weight tying

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None, prev_hiddens=None, prev_cells=None, return_prev_hiddens=False):
        """
        Forward pass of the RNN.

        Args:
            idx (torch.LongTensor): Input indices of shape (batch_size, sequence_length).
            targets (torch.LongTensor, optional): Target indices.
            prev_hiddens (torch.Tensor, optional): Previous hidden states.
            prev_cells (torch.Tensor, optional): Previous cell states (for LSTM only).
            return_prev_hiddens (bool, optional): Whether to return the final hidden states.

        Returns:
            Various combinations of logits, loss, and hidden states based on cell type and return_prev_hiddens.
        """
        device = idx.device
        b, t = idx.size()

        # Token and positional embeddings
        emb = self.wte(idx)
        #pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0).expand(b, t)
        #pos_emb = self.wpe(pos)
        emb = emb# + pos_emb

        # Initialize hidden states and cell states
        if prev_hiddens is not None:
            #if prev_hiddens.size(0) != self.n_layer or prev_hiddens.size(1) != b or prev_hiddens.size(2) != self.config.n_embd:
            #    raise ValueError(f"prev_hiddens has invalid shape: {prev_hiddens.shape}")
            hprev = prev_hiddens
        else:
            hprev = self.start.expand(-1, b, -1).contiguous()

        # Initialize cell states for LSTM
        if self.type == 'lstm':
            if prev_cells is not None:
                if prev_cells.size(0) != self.n_layer or prev_cells.size(1) != b or prev_cells.size(2) != self.config.n_embd:
                    raise ValueError(f"prev_cells has invalid shape: {prev_cells.shape}")
                cprev = prev_cells
            else:
                cprev = self.start_ct.expand(-1, b, -1).contiguous()

        # Prepare dropout masks
        variational_dropout_masks = []
        if self.training and self.variational_dropout_prob > 0:
            for l in range(self.n_layer):
                mask = torch.bernoulli(torch.full((b, self.config.n_embd), 1 - self.variational_dropout_prob, device=device)) / (1 - self.variational_dropout_prob)
                variational_dropout_masks.append(mask)
        else:
            variational_dropout_masks = [None] * self.n_layer

        zoneout_masks = []
        if self.training and self.zoneout_prob > 0:
            for l in range(self.n_layer):
                mask = torch.bernoulli(torch.full((b, self.config.n_embd), self.zoneout_prob, device=device))
                zoneout_masks.append(mask)
        else:
            zoneout_masks = [None] * self.n_layer

        hiddens = []
        cells = [] if self.type == 'lstm' else None

        for i in range(t):
            xt = emb[:, i, :]
            new_hprev = []
            new_cprev = [] if self.type == 'lstm' else None

            for l, cell in enumerate(self.cells):
                if self.type == 'lstm':
                    ht, ct = cell(xt, (hprev[l], cprev[l]))
                    new_cprev.append(ct)
                elif self.type == 'rru':
                    ht, _ = cell(xt, hprev[l])
                else:
                    #print(hprev[l])
                    ht = cell(xt, hprev[l])

                # Apply layer normalization and dropout
                ht = self.layer_norms[l](ht)
                ht = self.dropout(ht)

                # Apply Variational Dropout
                if variational_dropout_masks[l] is not None:
                    ht = ht * variational_dropout_masks[l]

                # Apply Zoneout
                if zoneout_masks[l] is not None:
                    ht = (1 - zoneout_masks[l]) * ht + zoneout_masks[l] * hprev[l]

                new_hprev.append(ht)
                xt = ht + xt  # Residual connection

            hprev = torch.stack(new_hprev)
            if self.type == 'lstm':
                cprev = torch.stack(new_cprev)
                cells.append(cprev[-1])
            hiddens.append(hprev[-1])

        # Stack all hidden states from the last layer
        hidden = torch.stack(hiddens, dim=1)
        logits = self.lm_head(hidden)

        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

            # Apply AR and TAR regularization if enabled
            if self.training and (self.ar_alpha > 0 or self.tar_beta > 0):
                AR_loss = hidden.pow(2).mean()
                if t > 1:
                    diff = hidden[:, 1:, :] - hidden[:, :-1, :]
                    TAR_loss = diff.pow(2).mean()
                else:
                    TAR_loss = 0.0
                loss = loss + self.ar_alpha * AR_loss + self.tar_beta * TAR_loss

        if return_prev_hiddens:
            if self.type in ['lstm']:
                # Return both hidden and cell states for LSTM
                return logits, hprev, cprev, loss
            else:
                # Return only hidden states for other cell types
                return logits, hprev, loss
        else:
            return logits, loss




# -----------------------------------------------------------------------------
# MLP language model

class TrainableLinearActivation(nn.Module):
    def __init__(self, num_neurons):
        super(TrainableLinearActivation, self).__init__()
        self.gamma = nn.Parameter(torch.FloatTensor(num_neurons).uniform_(-1, 1))

    def forward(self, x):
        return self.gamma * x

class MLPOG_GELU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size + 1, config.n_embd)

        # Adjust the first and last layers of the MLP to match input/output dimensions
        mlp_layers = [
            #nn.LayerNorm(self.block_size * config.n_embd),
            #MVP(self.block_size * config.n_embd, 50),
            nn.Linear(self.block_size * config.n_embd, config.n_embd),
            #nn.SiLU(),
            nn.LayerNorm(config.n_embd),
            MVP(config.n_embd, 256)
        ]
        for _ in range(config.n_layer - 1):  # One less because the first layer is already defined
            mlp_layers.extend([
                #MVP(config.n_embd, 50),
                nn.Linear(config.n_embd, config.n_embd),
                #nn.SiLU(),
                #Rational()
                nn.LayerNorm(config.n_embd),
                MVP(config.n_embd, 256)
            ])
        mlp_layers.append(nn.Linear(config.n_embd, self.vocab_size))  # Final layer to output vocab_size
        mlp_layers.append(nn.LayerNorm(self.vocab_size))
        mlp_layers.append(MVP(self.vocab_size, 256))
        self.mlp = nn.Sequential(*mlp_layers)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        embs = []
        for k in range(self.block_size):
            tok_emb = self.wte(idx)
            idx = torch.roll(idx, 1, 1)
            idx[:, 0] = self.vocab_size
            embs.append(tok_emb)

        x = torch.cat(embs, -1)
        logits = self.mlp(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss
class MLPOG(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size + 1, config.n_embd)

        # Adjust the first and last layers of the MLP to match input/output dimensions
        mlp_layers = [
            nn.Linear(self.block_size * config.n_embd, config.n_embd),
            nn.Sigmoid()
        ]
        for _ in range(config.n_layer - 1):  # One less because the first layer is already defined
            mlp_layers.extend([
                nn.Linear(config.n_embd, config.n_embd),
                nn.Sigmoid()
                #MLPBlock(config)
            ])
        mlp_layers.append(nn.Linear(config.n_embd, self.vocab_size))  # Final layer to output vocab_size
        self.mlp = nn.Sequential(*mlp_layers)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        embs = []
        for k in range(self.block_size):
            tok_emb = self.wte(idx)
            idx = torch.roll(idx, 1, 1)
            idx[:, 0] = self.vocab_size
            embs.append(tok_emb)

        x = torch.cat(embs, -1)
        logits = self.mlp(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

class ADALIN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size + 1, config.n_embd)

        # Adjust the first and last layers of the MLP to match input/output dimensions
        mlp_layers = [
            nn.Linear(self.block_size * config.n_embd, config.n_embd),
        ]
        for _ in range(config.n_layer - 1):  # One less because the first layer is already defined
            mlp_layers.extend([
                nn.Linear(config.n_embd, config.n_embd),
                #MLPBlock(config)
            ])
        mlp_layers.append(nn.Linear(config.n_embd, self.vocab_size))  # Final layer to output vocab_size
        self.mlp = nn.Sequential(*mlp_layers)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        embs = []
        for k in range(self.block_size):
            tok_emb = self.wte(idx)
            idx = torch.roll(idx, 1, 1)
            idx[:, 0] = self.vocab_size
            embs.append(tok_emb)

        x = torch.cat(embs, -1)
        logits = self.mlp(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * (self.gamma + 1)

class MLPLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size + 1, config.n_embd)

        # Initialize MLP layers with residual connections and RMSNorm
        self.layers = nn.ModuleList()
        self.input_proj = nn.Linear(self.block_size * config.n_embd, config.n_embd)  # Input projection
        self.output_proj = nn.Linear(config.n_embd, self.vocab_size)  # Final projection
        self.act = nn.SiLU()

        for _ in range(config.n_layer - 1):  # Define intermediate layers
            self.layers.append(nn.Sequential(
                nn.LayerNorm(config.n_embd),
                nn.Linear(config.n_embd, config.n_embd),
                nn.SiLU()
            ))

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        embs = []
        for k in range(self.block_size):
            tok_emb = self.wte(idx)
            idx = torch.roll(idx, 1, 1)
            idx[:, 0] = self.vocab_size
            embs.append(tok_emb)

        x = torch.cat(embs, -1)
        x = self.act(self.input_proj(x))  # Initial projection

        # Apply layers with RMSNorm and residual connections
        for layer in self.layers:
            x = x + layer(x)

        logits = self.output_proj(x)  # Final projection to output logits

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Custom complex linear layer


# Modified MLPLLM class with complex hidden layers
class ComplexMLP(nn.Module):
    def __init__(self, config):
        super(ComplexMLP, self).__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size + 1, config.n_embd)

        # Initialize MLP layers with residual connections and ComplexLayerNorm
        self.layers = nn.ModuleList()
        self.input_proj = ComplexLinear(self.block_size * config.n_embd, config.n_embd)  # Complex input projection

        # Output projection will handle real output from complex hidden states
        self.output_proj = nn.Linear(2 * config.n_embd, self.vocab_size)  # Final projection

        for _ in range(config.n_layer - 1):  # Define intermediate layers
            self.layers.append(nn.Sequential(
                ComplexLayerNorm(config.n_embd),
                ComplexLinear(config.n_embd, config.n_embd),
                ComplexSiLU(config.n_embd)
            ))

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        embs = []
        for _ in range(self.block_size):
            tok_emb = self.wte(idx)  # Real-valued embeddings
            tok_emb = torch.complex(tok_emb, tok_emb)  # Convert to complex
            idx = torch.roll(idx, 1, dims=1)
            idx[:, 0] = self.vocab_size
            embs.append(tok_emb)

        x = torch.cat(embs, dim=-1)
        x = self.input_proj(x)  # Complex input projection

        # Apply complex layers with residual connections
        for layer in self.layers:
            x = x + layer(x)

        # Prepare for real output projection
        x_real = torch.cat([x.real, x.imag], dim=-1)  # Concatenate real and imaginary parts
        logits = self.output_proj(x_real)  # Real-valued logits

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

class ParabolicMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size + 1, config.n_embd)

        # Adjust the first and last layers of the MLP to match input/output dimensions
        mlp_layers = [
            nn.Linear(self.block_size * config.n_embd, config.n_embd),
            RMSNorm(config.n_embd),
            ParabolicConeActivation(config.n_embd)#nn.SiLU()
        ]
        for _ in range(config.n_layer - 1):  # One less because the first layer is already defined
            mlp_layers.extend([
                nn.Linear(config.n_embd, config.n_embd),
                RMSNorm(config.n_embd),
                ParabolicConeActivation(config.n_embd)#nn.SiLU()
                #MLPBlock(config)
            ])
        mlp_layers.append(nn.Linear(config.n_embd, self.vocab_size))  # Final layer to output vocab_size
        self.mlp = nn.Sequential(*mlp_layers)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        embs = []
        for k in range(self.block_size):
            tok_emb = self.wte(idx)
            idx = torch.roll(idx, 1, 1)
            idx[:, 0] = self.vocab_size
            embs.append(tok_emb)

        x = torch.cat(embs, -1)
        logits = self.mlp(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

class MLPBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act1     = MVP(4 * config.n_embd, 50),
            actin     = MVP(config.n_embd, 50),
            actout     = MVP(config.n_embd, 50),
        ))
        m = self.mlp
        self.pau1 = MVP(config.n_embd, 50)
        self.pau2 = MVP(config.n_embd, 50)
        self.mlpf = lambda x: m.actout(m.c_proj(m.act1(m.c_fc(m.actin(x))))) # MLP forward

    def forward(self, x):
        #x = self.ln_1(x)
        x = x + self.mlpf(self.ln_2(x))
        return x

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size + 1, config.n_embd)

        # Adjust the first and last layers of the MLP to match input/output dimensions
        mlp_layers = [
            #MVP(self.block_size * config.n_embd, 50),
            nn.Linear(self.block_size * config.n_embd, config.n_embd),
            MVP(config.n_embd, 50)
        ]
        for _ in range(config.n_layer - 1):  # One less because the first layer is already defined
            mlp_layers.extend([
                #nn.Linear(config.n_embd, config.n_embd),
                #NewGELU()
                MLPBlock(config)
            ])
        mlp_layers.append(nn.Linear(config.n_embd, self.vocab_size))  # Final layer to output vocab_size
        mlp_layers.append(MVP(self.vocab_size, 50))  # Final layer to output vocab_size
        self.mlp = nn.Sequential(*mlp_layers)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        embs = []
        for k in range(self.block_size):
            tok_emb = self.wte(idx)
            idx = torch.roll(idx, 1, 1)
            idx[:, 0] = self.vocab_size
            embs.append(tok_emb)

        x = torch.cat(embs, -1)
        logits = self.mlp(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

# https://arxiv.org/abs/2410.01201v1

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Identity, Module
from torch.jit import script

def exists(v):
    return v is not None

# appendix B
# https://github.com/glassroom/heinsen_sequence
@script
def heinsen_associative_scan_log(log_coeffs, log_values):
    a_star = log_coeffs.cumsum(dim = 1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim = 1)
    log_h = a_star + log_h0_plus_b_star
    return log_h.exp()

# appendix B.3
@script
def g(x):
    return torch.where(x >= 0, x + 0.5, x.sigmoid())
@script
def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

# log-space version of minGRU - B.3.1
# they enforce the hidden states to be positive
import torch
import torch.nn as nn
import torch
import torch.nn as nn

class CubicActivation(nn.Module):
    def __init__(self, num_features):
        super(CubicActivation, self).__init__()
        self.alpha = nn.Parameter(torch.full((num_features,), 1.0))
        self.beta = nn.Parameter(torch.full((num_features,), 1.0))
        self.delta = nn.Parameter(torch.full((num_features,), 1.0))
        self.gamma = nn.Parameter(torch.full((num_features,), 1.0))
        self.sigma = nn.Parameter(torch.full((num_features,), 1.0))
    
    def forward(self, x):
        x = x * self.sigma
        # Cubic function: f(x) = α + βx - δx² + x³
        return self.gamma * (self.alpha + self.beta * x - self.delta * x**2 + x**3)

class GLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GLU, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)
        self.act = ParabolicConeActivation(output_dim)

    def forward(self, x):
        return self.linear(x) * self.act(self.gate(x))


class minGRU(Module):
    def __init__(self, dim, expansion_factor = 1.):
        super().__init__()

        dim_inner = int(dim * expansion_factor)
        self.to_hidden_and_gate = Linear(dim, dim_inner * 2, bias = False)
        self.to_out = Linear(dim_inner, dim, bias = False)# if expansion_factor != 1. else Identity()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.act1 = nn.Sigmoid()

    def forward(self, x, prev_hidden = None, return_next_prev_hidden = False):
        seq_len = x.shape[1]
        hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim = -1)

        if seq_len == 1:
            # handle sequential

            hidden = g(hidden)
            gate = self.act1(gate)#.sigmoid()
            out = torch.lerp(prev_hidden, hidden, gate) if exists(prev_hidden) else (hidden * gate)
        else:
            # parallel

            log_coeffs = -F.softplus(gate)

            log_z = -F.softplus(-gate)
            log_tilde_h = log_g(hidden)
            log_values = log_z + log_tilde_h

            if exists(prev_hidden):
                log_values = torch.cat((log_g(prev_hidden), log_values), dim = 1)
                log_coeffs = F.pad(log_coeffs, (0, 0, 1, 0))

            out = heinsen_associative_scan_log(log_coeffs, log_values)
            out = out[:, -seq_len:]

        next_prev_hidden = out[:, -1:]

        out = self.to_out(out)

        if not return_next_prev_hidden:
            return out# * self.gamma.clamp(min=0, max=1)

        return out, next_prev_hidden# * self.gamma, next_prev_hidden



class AHAF(torch.nn.Module):
    def __init__(self, *, size: Tuple[int, ...] = (1,), init_as: str = 'ReLU'):
        super(AHAF, self).__init__()

        if init_as == 'ReLU':
            self.gamma = torch.nn.Parameter(torch.ones(*size)*1e9)
            self.beta = torch.nn.Parameter(torch.ones(*size))
        elif init_as == 'SiL':
            self.gamma = torch.nn.Parameter(torch.ones(*size))
            self.beta = torch.nn.Parameter(torch.ones(*size))
        elif init_as == 'CUSTOM':
            self.gamma = torch.nn.Parameter(torch.ones(*size)*10)
            self.beta = torch.nn.Parameter(torch.ones(*size))
        else:
            raise ValueError("Invalid initialization mode [{}]".format(init_as))

    def forward(self, inputs):
        sig_in = self.gamma * inputs
        sig_out = torch.sigmoid(sig_in)
        amplified = inputs * self.beta
        out = sig_out * amplified
        return out

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d
class CapActi(nn.Module):
    def __init__(self, num_parameters, lower_cap_init=-1, upper_cap_init=1):
        super(CapActi, self).__init__()
        self.lower_cap = nn.Parameter(torch.full((num_parameters,), lower_cap_init, dtype=torch.float32))
        self.upper_cap = nn.Parameter(torch.full((num_parameters,), upper_cap_init, dtype=torch.float32))

    def forward(self, x):
        x = torch.min(torch.max(x, self.lower_cap), self.upper_cap)
        return x
# classes
class SRSFunction(nn.Module):
    def __init__(self, num_features, alpha_init=3.0, beta_init=2.0, trainable_alpha=True, trainable_beta=True):
        super(SRSFunction, self).__init__()
        # Initialize alpha and beta for each feature
        self.alpha = nn.Parameter(torch.full((num_features,), alpha_init), requires_grad=trainable_alpha)
        self.beta = nn.Parameter(torch.full((num_features,), beta_init), requires_grad=trainable_beta)

    def forward(self, x):
        # Ensure alpha and beta are broadcastable to match the input shape
        numerator = x
        denominator = (x / self.alpha) + torch.exp(-x / self.beta)
        return numerator / denominator


class CustomActivation(nn.Module):
    def __init__(self, input_size):
        super(CustomActivation, self).__init__()
        # Initialize beta as a trainable parameter for each feature
        self.beta = nn.Parameter(torch.ones(1, input_size))

    def forward(self, x):
        # Compute beta * x element-wise
        beta_x = self.beta * x
        # Apply the activation function: tanh(x) * sin(beta * x)
        return torch.tanh(x) * torch.sin(beta_x)

import torch
import torch.nn as nn

class ComplexActivation(nn.Module):
    def __init__(self):
        super(ComplexActivation, self).__init__()

    def forward(self, x):
        return x * torch.tanh(torch.sin(x ** 2) * torch.exp(-x ** 2))
import torch
import torch.nn as nn

class ParametricCompositeActivation(nn.Module):
    def __init__(self, num_features, conv_compatible=False):
        super(ParametricCompositeActivation, self).__init__()
        self.conv_compatible = conv_compatible

        # Initialize parameters for each feature/channel
        self.a1 = nn.Parameter(torch.randn(num_features))
        self.b1 = nn.Parameter(torch.randn(num_features))
        self.shift_x1 = nn.Parameter(torch.zeros(num_features))  # Shift along x-axis for Tanh
        self.shift_y1 = nn.Parameter(torch.zeros(num_features))  # Shift along y-axis for Tanh
        self.cap1_min = nn.Parameter(torch.tensor(-4.0).expand(num_features))
        self.cap1_max = nn.Parameter(torch.tensor(4.0).expand(num_features))

        self.a2 = nn.Parameter(torch.randn(num_features))
        self.b2 = nn.Parameter(torch.randn(num_features))
        self.shift_x2 = nn.Parameter(torch.zeros(num_features))  # Shift along x-axis for Sine
        self.shift_y2 = nn.Parameter(torch.zeros(num_features))  # Shift along y-axis for Sine
        self.cap2_min = nn.Parameter(torch.tensor(-4.0).expand(num_features))
        self.cap2_max = nn.Parameter(torch.tensor(4.0).expand(num_features))

        self.a3 = nn.Parameter(torch.randn(num_features))
        self.b3 = nn.Parameter(torch.abs(torch.randn(num_features)))  # Ensure positive for Gaussian width
        self.shift_x3 = nn.Parameter(torch.zeros(num_features))  # Shift along x-axis for Gaussian
        self.shift_y3 = nn.Parameter(torch.zeros(num_features))  # Shift along y-axis for Gaussian
        self.cap3_min = nn.Parameter(torch.tensor(-4.0).expand(num_features))
        self.cap3_max = nn.Parameter(torch.tensor(4.0).expand(num_features))

        self.a4 = nn.Parameter(torch.randn(num_features))
        self.shift_x4 = nn.Parameter(torch.zeros(num_features))  # Shift along x-axis for Cubic
        self.shift_y4 = nn.Parameter(torch.zeros(num_features))  # Shift along y-axis for Cubic
        self.cap4_min = nn.Parameter(torch.tensor(-4.0).expand(num_features))
        self.cap4_max = nn.Parameter(torch.tensor(4.0).expand(num_features))
        self.pau = PAU(m=6, n=5)

    def forward(self, x):
        if self.conv_compatible:
            # Broadcasting for conv compatibility
            x1 = torch.clamp(x - self.shift_x1[..., None, None], self.cap1_min[..., None, None], self.cap1_max[..., None, None])
            x2 = torch.clamp(x - self.shift_x2[..., None, None], self.cap2_min[..., None, None], self.cap2_max[..., None, None])
            x3 = torch.clamp(x - self.shift_x3[..., None, None], self.cap3_min[..., None, None], self.cap3_max[..., None, None])
            x4 = torch.clamp(x - self.shift_x4[..., None, None], self.cap4_min[..., None, None], self.cap4_max[..., None, None])
            term1 = (self.a1[..., None, None] * torch.tanh(self.b1[..., None, None] * x1)) + self.shift_y1[..., None, None]
            term2 = (self.a2[..., None, None] * torch.sin(self.b2[..., None, None] * x2)) + self.shift_y2[..., None, None]
            term3 = (self.a3[..., None, None] * torch.exp(-self.b3[..., None, None] * x3 ** 2)) + self.shift_y3[..., None, None]
            term4 = (self.a4[..., None, None] * x4 ** 3) + self.shift_y4[..., None, None]
        else:
            # No broadcasting required for dense layers
            x1 = torch.clamp(x, self.cap1_min, self.cap1_max) # - self.shift_x1
            x2 = torch.clamp(x, self.cap2_min, self.cap2_max)
            x3 = torch.clamp(x, self.cap3_min, self.cap3_max)
            x4 = torch.clamp(x, self.cap4_min, self.cap4_max)
            term1 = (self.a1 * torch.tanh(self.b1 * x1))# + self.shift_y1
            term2 = (self.a2 * torch.sin(self.b2 * x2))# + self.shift_y2
            term3 = (self.a3 * torch.exp(-self.b3 * x3 ** 2))# + self.shift_y3
            term4 = (self.a4 * x4 ** 3)# + self.shift_y4

        # Compute each term with y-axis shift for each feature/channel

        return term1 + term2 + term3 + term4




import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

class ParametricELU(nn.Module):
    def __init__(self, num_features):
        super(ParametricELU, self).__init__()
        # Learnable alpha and beta for each feature
        self.alpha = nn.Parameter(torch.ones(num_features))  # Default to 1 for ELU behavior
        self.beta = nn.Parameter(torch.ones(num_features))   # Default to 1 for ELU behavior

    def forward(self, x):
        # Apply PELU formula for each feature
        positive = F.relu(x) * self.beta
        negative = (self.alpha * (torch.exp(x / self.alpha) - 1)) * (x < 0).float()
        return positive + negative

class THAF1(nn.Module):
    def __init__(self, input_size):
        super(THAF1, self).__init__()
        self.W_r = nn.Parameter(torch.randn(input_size))
        self.W_i = nn.Parameter(torch.randn(input_size))
        self.W_j = nn.Parameter(torch.randn(input_size))
        self.W_k = nn.Parameter(torch.randn(input_size))
        self.activation1 = ParametricCompositeActivation()  # You can choose other activations
        self.activation2 = ParametricCompositeActivation()  # You can choose other activations
        self.activation3 = ParametricCompositeActivation()  # You can choose other activations
        self.activation4 = ParametricCompositeActivation()  # You can choose other activations

    def forward(self, x):
        x_r = self.activation1(self.W_r * x)
        x_i = self.activation2(self.W_i * x)
        x_j = self.activation3(self.W_j * x)
        x_k = self.activation4(self.W_k * x)
        
        # Compute the modulus of the quaternion
        y = torch.sqrt(x_r**2 + x_i**2 + x_j**2 + x_k**2)
        return y

class THAF(nn.Module):
    def __init__(self, input_size):
        super(THAF, self).__init__()
        self.W_r = nn.Parameter(torch.randn(input_size))
        self.W_i = nn.Parameter(torch.randn(input_size))
        self.W_j = nn.Parameter(torch.randn(input_size))
        self.W_k = nn.Parameter(torch.randn(input_size))
        self.activation1 = THAF1(input_size)  # You can choose other activations
        self.activation2 = THAF1(input_size)  # You can choose other activations
        self.activation3 = THAF1(input_size)  # You can choose other activations
        self.activation4 = THAF1(input_size)  # You can choose other activations

    def forward(self, x):
        x_r = self.activation1(self.W_r * x)
        x_i = self.activation2(self.W_i * x)
        x_j = self.activation3(self.W_j * x)
        x_k = self.activation4(self.W_k * x)
        
        # Compute the modulus of the quaternion
        y = torch.sqrt(x_r**2 + x_i**2 + x_j**2 + x_k**2)
        return y
class SReLU(nn.Module):
    def __init__(self, num_parameters=1, init_a_l=0.5, init_a_r=0.5, init_t_l=0.1, init_t_r=1.0):
        super(SReLU, self).__init__()
        self.num_parameters = num_parameters

        # Initialize thresholds t_l and t_r
        self.t_l = nn.Parameter(torch.full((num_parameters,), init_t_l))
        delta_t_init = init_t_r - init_t_l  # Initial difference between t_r and t_l
        self.delta_t = nn.Parameter(torch.full((num_parameters,), delta_t_init))

        # Initialize slopes a_l and a_r
        self.a_l_raw = nn.Parameter(torch.full((num_parameters,), init_a_l))
        self.a_r_raw = nn.Parameter(torch.full((num_parameters,), init_a_r))

    def forward(self, x):
        # Reshape parameters for broadcasting
        shape = [1, -1] + [1] * (x.dim() - 2)

        t_l = self.t_l.view(shape)
        delta_t = F.softplus(self.delta_t.view(shape))
        t_r = t_l + delta_t  # Ensure t_r >= t_l

        a_l = F.softplus(self.a_l_raw.view(shape))  # Ensure slopes are positive
        a_r = F.softplus(self.a_r_raw.view(shape))

        # Compute masks for different regions
        mask_l = x <= t_l
        mask_r = x >= t_r

        # Compute outputs for each region
        y_l = t_l + a_l * (x - t_l)
        y_r = t_r + a_r * (x - t_r)
        y_c = x.clone()  # Central region where output equals input

        # Combine outputs based on masks
        y = torch.where(mask_l, y_l, y_c)
        y = torch.where(mask_r, y_r, y)

        return y
class EnhancedAdaptiveRationalActivation(nn.Module):
    def __init__(self, dim):
        super(EnhancedAdaptiveRationalActivation, self).__init__()
        # Initialize learnable parameters
        self.dim = dim
        self.alpha = nn.Parameter(torch.ones(dim))   # Controls amplitude
        self.beta = nn.Parameter(torch.ones(dim))    # Controls curvature
        self.gamma = nn.Parameter(torch.ones(dim))   # Controls input scaling

    def forward(self, x):
        scaled_x = self.gamma * x
        return self.alpha * x * (scaled_x / torch.sqrt(1 + self.beta * scaled_x**2))
import torch
import torch.nn as nn

class ASwish(nn.Module):
    def __init__(self, dim=1):
        super(ASwish, self).__init__()
        # Initialize beta as a trainable parameter
        self.beta = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Apply ASwish with trainable beta
        return x * torch.sigmoid(self.beta * x)

import torch
import torch.nn as nn
import torch
import torch.nn as nn

class CubicActivationa(nn.Module):
    def __init__(self, num_features):
        super(CubicActivationa, self).__init__()
        self.alpha = nn.Parameter(torch.full((num_features,), 1.0))
        self.beta = nn.Parameter(torch.full((num_features,), 1.0))
        self.delta = nn.Parameter(torch.full((num_features,), 1.0))
        self.gamma = nn.Parameter(torch.full((num_features,), 1.0))
        self.sigma = nn.Parameter(torch.full((num_features,), 1.0))
    
    def forward(self, x):
        x = x * self.sigma
        # Cubic function: f(x) = α + βx - δx² + x³
        return self.gamma * (self.alpha + self.beta * x - self.delta * x**2 + x**3)
import torch
import torch.nn as nn

class CubicActivation(nn.Module):
    def __init__(self, num_features):
        super(CubicActivation, self).__init__()
        self.alpha = nn.Parameter(torch.full((num_features,), 1.0))
        self.beta = nn.Parameter(torch.full((num_features,), 2.0))
        self.delta = nn.Parameter(torch.full((num_features,), 1.0))
        self.gamma = nn.Parameter(torch.full((num_features,), 1.0))
        self.sigma = nn.Parameter(torch.full((num_features,), 1.0))
        self.omega = nn.Parameter(torch.full((num_features,), 1.0))

    def forward(self, x):
        x = x * self.sigma
        return self.omega * (self.gamma + x * (self.alpha + x * (self.beta + x * (self.delta - x))))
class ProtoAct(nn.Module):
    def __init__(self, num_features):
        super(ProtoAct, self).__init__()
        self.alpha = nn.Parameter(torch.full((num_features,), 1.0))
        self.beta = nn.Parameter(torch.full((num_features,), 2.0))
        self.delta = nn.Parameter(torch.full((num_features,), 1.0))
        self.gamma = nn.Parameter(torch.full((num_features,), 1.0))
        self.sigma = nn.Parameter(torch.full((num_features,), 1.0))
        self.omega = nn.Parameter(torch.full((num_features,), 1.0))
        self.theta = nn.Parameter(torch.full((num_features,), 1.0))
        self.lamda = nn.Parameter(torch.full((num_features,), 1.0))

    def forward(self, x):
        x = x * self.sigma
        return self.lamda * (self.theta + x * (self.omega + x * (self.gamma + x * (self.alpha + x * (self.beta + x * (self.delta - x))))))
class ConeActivation(nn.Module):
    def forward(self, x):
        return 1 - torch.abs(x - 1)
class ParameterizedConeActivation(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, learnable=True):
        super().__init__()
        if learnable:
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
            self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))
            self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        else:
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma

    def forward(self, x):
        return self.beta - torch.abs(x - self.gamma) ** self.alpha
class Sqish(nn.Module):
    def __init__(self, a=0.5, beta=1.0, gamma=1.0, trainable=True):
        super(Sqish, self).__init__()
        if trainable:
            # Parameters are initialized and set as trainable
            self.a = nn.Parameter(torch.tensor(a, dtype=torch.float32))
            self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32))
            self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))
        else:
            # Parameters are fixed and registered as buffers (non-trainable)
            self.register_buffer('a', torch.tensor(a, dtype=torch.float32))
            self.register_buffer('beta', torch.tensor(beta, dtype=torch.float32))
            self.register_buffer('gamma', torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x):
        a = self.a
        beta = self.beta
        gamma = self.gamma
        term = (1 - a) * x
        denominator = 1 + beta * torch.exp(-2 * gamma * term)
        return a * x + term / denominator

import torch
import torch.nn as nn

class PEUAF(nn.Module):
    def __init__(self, init_w=1.0):
        super(PEUAF, self).__init__()
        # Initialize the frequency parameter 'w' as a learnable parameter
        self.w = nn.Parameter(torch.tensor(init_w))

    def forward(self, x):
        w = self.w
        # Split x into positive and negative parts
        x_positive = torch.relu(x)  # x if x >= 0 else 0
        x_negative = torch.clamp(x, max=0.0)  # x if x <= 0 else 0

        # Compute PEUAF for x >= 0
        s = w * x_positive
        # s mod 2
        s_mod_2 = torch.fmod(s, 2.0)
        # PEUAF(x) = |w*x - 2 * floor(w*x + 1/2)|
        # Simplified as: y_positive = 1 - abs((s mod 2) - 1)
        y_positive = torch.abs(s_mod_2 - 2 * torch.floor((s + 0.5) / 2))

        # Compute PEUAF for x < 0
        y_negative = x_negative / (1.0 + torch.abs(x_negative))

        # Combine the positive and negative parts
        y = y_positive + y_negative
        return y
import torch
import torch.nn as nn
import torch.nn.functional as F

class ModSwish(nn.Module):
    def forward(self, x):
        return x / (1 - x * torch.exp(-x))

class PRSigELU(nn.Module):
    def __init__(self, alpha_init=1.0, beta_init=1.0):
        super(PRSigELU, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.beta = nn.Parameter(torch.tensor(beta_init))

    def forward(self, x):
        return torch.where(
            x >= 0,
            self.alpha * x,
            self.beta * (torch.sigmoid(x) - 1)
        )
import torch
import torch.nn as nn
import numpy as np

class MultiThresholdReLU(nn.Module):
    def __init__(self, thresholds, scales):
        """
        Multi-Threshold ReLU activation function.
        
        Parameters:
            thresholds (list of floats): Points at which the activation function's slope changes.
            scales (list of floats): Scaling factors for each ReLU segment.
        """
        super(MultiThresholdReLU, self).__init__()
        assert len(thresholds) == len(scales), "Thresholds and scales must have the same length."
        self.thresholds = torch.tensor(thresholds)
        self.scales = torch.tensor(scales)

    def forward(self, x):
        out = torch.zeros_like(x)
        for t, s in zip(self.thresholds, self.scales):
            out += s * torch.relu(x - t)
        return out


import torch
import torch.nn as nn
import torch.nn.functional as F

class APALU(nn.Module):
    def __init__(self, a_init=0.55, b_init=0.065):
        super(APALU, self).__init__()
        # Initialize a and b as trainable parameters
        self.a = nn.Parameter(torch.tensor(a_init))
        self.b = nn.Parameter(torch.tensor(b_init))

    def forward(self, x):
        # Define the APALU function based on the conditions
        pos = self.a * (x + x / (1 + torch.exp(-1.702 * x)))
        neg = self.b * (torch.exp(x) - 1)
        return torch.where(x >= 0, pos, neg)


def FeedForward(dim, mult = 4):
    dim_inner = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, dim_inner),
        nn.SiLU(),#ParabolicConeActivation(dim),#SRSFunction(1, 1., 1.),
        nn.Linear(dim_inner, dim)
    )

# conv

class CausalDepthWiseConv1d(Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size = kernel_size, groups = dim),
            nn.Conv1d(dim, dim, kernel_size = 1)
        )
    def forward(self, x):
        x = x.transpose(1, 2) # b n d -> b d n
        x = F.pad(x, (self.kernel_size - 1, 0), value = 0.)
        x = self.net(x)
        return x.transpose(1, 2) # b d n -> b n d

# main class

class minGRULM(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        ff_mult=1.5,
        min_gru_expansion=1.5,
        blocks=1,
        conv_kernel_size=3,
        enable_conv=False
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.blocks = blocks

        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                CausalDepthWiseConv1d(dim, conv_kernel_size) if enable_conv else None,
                RMSNorm(dim),
                minGRU(dim, expansion_factor=min_gru_expansion),
                RMSNorm(dim),
                FeedForward(dim, mult=ff_mult)
            ]))

        self.norm = RMSNorm(dim)
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)

    def get_block_size(self):
        return self.blocks  # This model only needs one previous character to predict the next

    def forward(
        self,
        x,
        targets=None,
        return_prev_hiddens=False,
        prev_hiddens=None
    ):
        if targets is not None:
            x, labels = x[:, :-1], x[:, 1:]

        x = self.token_emb(x)

        # Handle previous hiddens for recurrent decoding
        if exists(prev_hiddens):
            x = x[:, -1:]

        next_prev_hiddens = []
        prev_hiddens = iter(default(prev_hiddens, []))

        for conv, norm, mingru, ff_norm, ff in self.layers:
            # Convolution
            if exists(conv):
                x = conv(x) + x

            # minGRU
            prev_hidden = next(prev_hiddens, None)
            min_gru_out, next_prev_hidden = mingru(
                norm(x),
                prev_hidden,
                return_next_prev_hidden=True
            )
            x = min_gru_out + x
            next_prev_hiddens.append(next_prev_hidden)
            #x = ff(ff_norm(x)) + x

        # Final normalization and projection to logits
        embed = self.norm(x)
        logits = self.to_logits(embed)

        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),  # Flatten logits for all tokens in the batch
                labels.reshape(-1),                   # Flatten labels
                ignore_index=-1                    # Optional: ignore padding index if needed
            )

        if not return_prev_hiddens:
            return logits, loss

        return logits, next_prev_hiddens, loss



# -----------------------------------------------------------------------------
# Bigram language model

class Bigram(nn.Module):
    """
    Bigram Language Model 'neural net', simply a lookup table of logits for the
    next character given a previous character.
    """

    def __init__(self, config):
        super().__init__()
        n = config.vocab_size
        self.logits = nn.Parameter(torch.zeros((n, n)))

    def get_block_size(self):
        return 1 # this model only needs one previous character to predict the next

    def forward(self, idx, targets=None):

         # 'forward pass', lol
        logits = self.logits[idx]

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

# Chomp1d removes extra padding introduced for causal convolutions
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        return x[:, :, :-self.chomp_size]

# TemporalBlock is a building block of TCN with residual connections and dilations
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, stride=1, dropout=0.2):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.SiLU()
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.SiLU()
        self.init_weights()
    
    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight.data)
        nn.init.kaiming_normal_(self.conv2.weight.data)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight.data)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# TemporalConvNet stacks multiple TemporalBlocks
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i  # Exponentially increasing dilations
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, dilation=dilation, dropout=dropout
            ))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Updated CharConvNet using TemporalConvNet
class CharConvNet(nn.Module):
    def __init__(self, config):
        super(CharConvNet, self).__init__()
        self.block_size = config.block_size
        self.vocab_size = config.vocab_size

        # Embedding layer
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Temporal Convolutional Network with state-of-the-art techniques
        num_channels = [config.n_embd] * config.n_layer  # Keeping channel size consistent
        self.tcn = TemporalConvNet(
            num_inputs=config.n_embd,
            num_channels=num_channels,
            kernel_size=3,
            dropout=getattr(config, 'dropout', 0.1)
        )

        # Final linear layer
        self.ln = nn.Linear(config.n_embd, self.vocab_size)

    def get_block_size(self):
        return self.block_size

    def forward(self, idx, targets=None):
        # Input embedding
        x = self.wte(idx)  # (batch_size, seq_len, n_embd)
        x = x.transpose(1, 2)  # (batch_size, n_embd, seq_len)

        # Apply Temporal Convolutional Network
        x = self.tcn(x)
        x = x.transpose(1, 2)  # (batch_size, seq_len, n_embd)

        # Compute logits
        logits = self.ln(x)  # (batch_size, seq_len, vocab_size)

        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

class HyperMixerModel(nn.Module):
    """Main model class that implements the HyperMixer architecture."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Model hyperparameters
        self.vocab_size = config.vocab_size
        self.block_size = config.block_size
        self.n_embd = config.n_embd
        self.n_layer = config.n_layer
        self.drop = 0.1  # Dropout rate
        
        # Input embedding
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop_emb = nn.Dropout(self.drop)
        
        # HyperMixer blocks
        self.blocks = nn.ModuleList([
            HyperMixerBlock(
                dim=config.n_embd,
                mlp_ratio=(0.5, 4.0),
                drop=self.drop,
                drop_path=0.1 * (i / (config.n_layer - 1)) if config.n_layer > 1 else 0
            ) for i in range(config.n_layer)
        ])
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(config.n_embd)
        
        # Output head
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def configure_optimizers(self, weight_decay, learning_rate):
        # Separate out parameters that should and shouldn't have weight decay
        decay = set()
        no_decay = set()
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn
                if 'bias' in pn or 'ln' in mn or 'ln_f' in mn:
                    no_decay.add(fpn)
                else:
                    decay.add(fpn)
        
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
        return optim_groups
    
    def get_block_size(self):
        return self.block_size
    
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        
        # Assert sequence length is within block size
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        
        # Get token embeddings and add positional embeddings
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb[:, :t, :]
        x = self.drop_emb(tok_emb + pos_emb)
        
        # Apply HyperMixer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply final layer norm and project to vocabulary
        x = self.ln_f(x)
        logits = self.head(x)
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss


import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output
@dataclass
class HyperMixerConfig:
    """Configuration class to store the configuration of a HyperMixer model."""
    vocab_size: int = 50257  # GPT-2 vocab size
    block_size: int = 1024   # Maximum sequence length
    n_embd: int = 768       # Embedding dimension
    n_hidden: int = 384     # Hidden dimension for hypernetwork MLPs
    n_layer: int = 12       # Number of HyperMixer blocks
    drop: float = 0.1       # Dropout rate
    drop_path: float = 0.1  # Drop path rate

class HyperMixer(nn.Module):
    """Improved HyperMixer implementation with better gradient flow and token mixing."""
    
    def __init__(self, dim, hidden_dim, drop=0.):
        super().__init__()
        
        # Separate hypernetworks for input and output (but sharing some weights)
        self.shared_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(drop)
        )
        
        self.W1_net = nn.Linear(dim, hidden_dim)
        self.W2_net = nn.Linear(dim, hidden_dim)
        
        # Learnable scaling factors for mixing
        self.W1_scale = nn.Parameter(torch.ones(1))
        self.W2_scale = nn.Parameter(torch.ones(1))
        
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(drop)
        
        # Additional learnable weights for position integration
        self.pos_gate = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, pos_emb):
        B, N, D = x.shape
        
        # Gated position embedding integration
        pos_weight = self.pos_gate(x)
        token_info = x + pos_weight * pos_emb
        
        # Generate shared features
        shared_features = self.shared_net(token_info)
        
        # Generate W1 and W2 with learned scaling
        W1 = self.W1_scale * self.W1_net(shared_features)  # [B, N, hidden_dim]
        W2 = self.W2_scale * self.W2_net(shared_features)  # [B, N, hidden_dim]
        
        # Token mixing with improved gradient paths
        x_proj = torch.bmm(x.transpose(1, 2), W1)  # [B, D, hidden_dim]
        hidden = F.gelu(x_proj)
        output = torch.bmm(W2, hidden.transpose(1, 2))  # [B, N, D]
        
        # Normalize and regularize
        output = self.layer_norm(output)
        return self.drop(output)

class HyperMixerBlock(nn.Module):
    """Improved HyperMixer block with better regularization."""
    
    def __init__(self, dim, hidden_dim=None, drop=0., drop_path=0.):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim // 2
            
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        
        self.token_mixer = HyperMixer(dim=dim, hidden_dim=hidden_dim, drop=drop)
        
        # Improved channel mixing with gating
        self.channel_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
        self.channel_mixer = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(4 * dim, dim),
            nn.Dropout(drop)
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x, pos_emb):
        # Improved residual paths
        token_mix = self.token_mixer(self.norm1(x), pos_emb)
        x = x + self.drop_path(token_mix)
        
        # Gated channel mixing
        normed = self.norm2(x)
        gate = self.channel_gate(normed)
        channel_mix = self.channel_mixer(normed)
        x = x + self.drop_path(gate * channel_mix)
        
        return x

class HyperMixerModel(nn.Module):
    """Complete HyperMixer Language Model"""
    
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([
                HyperMixerBlock(
                    dim=config.n_embd,
                    hidden_dim=config.n_embd,
                    drop=0.0,
                    drop_path=0.1 * i / max(1, (config.n_layer - 1))
                ) for i in range(config.n_layer)
            ]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Report number of parameters
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print(f"number of parameters: {n_params/1e6:.2f}M")
    def get_block_size(self):
        return self.block_size - 1 # this model only needs one previous character to predict the next
        
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        
        # Get positional embeddings
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.transformer.wpe(pos)  # position embeddings (1, t, n_embd)
        
        # Forward pass
        tok_emb = self.transformer.wte(idx)  # token embeddings (b, t, n_embd)
        x = tok_emb
        
        # Pass through HyperMixer blocks with position embeddings
        for block in self.transformer.h:
            x = block(x, pos_emb)
            
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            
        return logits, loss


import torch
import torch.nn as nn
import torch.nn.functional as F

class MambaBlock(nn.Module):
    def __init__(self, dim, expansion_factor=2, d_state=16):
        super().__init__()
        d_inner = int(dim * expansion_factor)
        self.dim = dim
        self.d_inner = d_inner
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear_in = nn.Linear(dim, d_inner * 2, bias=False)
        self.mamba_ssm = SelectiveSSM(d_inner, d_state)
        self.linear_out = nn.Linear(d_inner, dim, bias=False)
        self.act = nn.SiLU()
        
    def forward(self, x):
        B, L, D = x.shape
        
        # First norm and project
        x_norm = self.norm1(x)
        x_proj = self.linear_in(x_norm)
        
        # Split into hidden and gate
        x_hidden, gate = x_proj.chunk(2, dim=-1)  # Each is [B, L, d_inner]
        
        # Apply SSM
        x_ssm = self.mamba_ssm(x_hidden)  # [B, L, d_inner]
        
        # Gate and project back
        x_out = self.linear_out(self.act(x_ssm) * gate)  # [B, L, dim]
        
        return x + x_out
        
class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.linear_delta = nn.Linear(d_model, 1, bias=False)
        self.linear_A = nn.Linear(d_model, d_state, bias=False)
        self.linear_B = nn.Linear(d_model, d_state, bias=False)
        # Change C to map from d_state to d_model
        self.linear_C = nn.Parameter(torch.randn(d_state, d_model) / math.sqrt(d_model))
        self.softplus = nn.Softplus()
        
    def forward(self, x, h = None):
        batch_size, seq_len, d_model = x.shape
        A = self.linear_A(x)  # [B, L, d_state]
        B_out = self.linear_B(x)  # [B, L, d_state]
        delta = self.softplus(self.linear_delta(x))  # [B, L, 1]
        
        if h is None:
            h = torch.zeros(batch_size, self.d_state, device=x.device, dtype=x.dtype)
            
        output = []
        h_t = h
        for t in range(seq_len):
            # Update hidden state
            h_t = (torch.exp(-delta[:,t]) * h_t) + (delta[:,t] * B_out[:,t])  # [B, d_state]
            # Generate output using fixed C matrix
            o_t = h_t @ self.linear_C  # [B, d_model]
            output.append(o_t)
            
        y = torch.stack(output, dim=1)  # [B, L, d_model]
        return y

class Mamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)

        self.layers = nn.ModuleList([
            MambaBlock(config.n_embd, expansion_factor=2, d_state=config.n_embd//4)
            for _ in range(config.n_layer)
        ])
        self.norm = nn.LayerNorm(config.n_embd)
        self.to_logits = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def get_block_size(self):
        return self.block_size

    def forward(self, x, targets=None):
        x = self.token_emb(x)
        for block in self.layers:
            x = block(x)
        x = self.norm(x)
        logits = self.to_logits(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss
