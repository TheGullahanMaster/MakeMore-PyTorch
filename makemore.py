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

from mymodels import *


# -----------------------------------------------------------------------------
# helper functions for evaluating and sampling from the model

import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
import numpy as np

class ActivationTracker:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.current_token_idx = 0
        self.activations_per_token = []
        self.all_tokens = []
        self.token_contributions = []
        self.previous_activations = {}
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def add_hooks(self):
        self.remove_hooks()
        
        def hook_fn(name):
            def forward_hook(module, input, output):
                if output.dim() == 3:
                    current_activation = output[0, -1].detach()
                elif output.dim() == 2:
                    current_activation = output[0].detach()
                else:
                    return

                while len(self.activations_per_token) <= self.current_token_idx:
                    self.activations_per_token.append({})
                    self.token_contributions.append({})

                self.activations_per_token[self.current_token_idx][name] = current_activation.cpu().numpy().tolist()

                if name in self.previous_activations:
                    prev_acts = self.previous_activations[name]
                    if len(prev_acts) > 0:
                        contributions = self.calculate_contributions(
                            torch.stack(prev_acts),
                            current_activation
                        )
                        self.token_contributions[self.current_token_idx][name] = contributions.cpu().numpy().tolist()

                if name not in self.previous_activations:
                    self.previous_activations[name] = []
                self.previous_activations[name].append(current_activation)

            return forward_hook
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm, nn.SiLU, nn.Tanh, nn.PReLU, nn.ReLU, nn.LeakyReLU, nn.Sigmoid, MVP, ParabolicConeActivation, nn.GELU)):
                hook = module.register_forward_hook(hook_fn(name))
                self.hooks.append(hook)
    
    def calculate_contributions(self, prev_activations, current_activation):
        if current_activation.dim() == 1:
            current_activation = current_activation.unsqueeze(0)
        
        prev_norm = F.normalize(prev_activations, dim=-1)
        curr_norm = F.normalize(current_activation, dim=-1)
        
        similarities = F.cosine_similarity(prev_norm, curr_norm)
        contributions = (similarities + 1) / 2
        
        return contributions
    
    def normalize_contributions(self, contributions, method='minmax', power=2):
        """
        Normalize contributions to enhance contrast.
        
        Args:
            contributions: List of contribution values
            method: 'minmax', 'softmax', or 'power'
            power: Exponent for power normalization
        """
        if not contributions:
            return contributions
            
        values = np.array(contributions)
        
        if method == 'minmax':
            # Enhanced min-max normalization
            min_val = np.min(values)
            max_val = np.max(values)
            if max_val > min_val:
                normalized = (values - min_val) / (max_val - min_val)
                # Apply power function to increase contrast
                normalized = np.power(normalized, power)
                return normalized.tolist()
            return contributions
            
        elif method == 'softmax':
            # Temperature-scaled softmax
            temperature = 0.1  # Lower temperature = more contrast
            exp_values = np.exp((values - np.max(values)) / temperature)
            normalized = exp_values / np.sum(exp_values)
            return normalized.tolist()
            
        elif method == 'power':
            # Direct power normalization
            normalized = np.power(values, power)
            normalized = normalized / np.max(normalized)
            return normalized.tolist()
            
        return contributions
    
    def save_activations(self, output_dir, norm_method='minmax', power=3):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        processed_contributions = []
        for token_idx in range(len(self.token_contributions)):
            token_contrib = {}
            for layer_name, contribs in self.token_contributions[token_idx].items():
                # Pad with zeros for missing previous tokens
                padded_contribs = [0.0] * token_idx
                if isinstance(contribs, list):
                    normalized_contribs = self.normalize_contributions(
                        contribs, 
                        method=norm_method,
                        power=power
                    )
                    padded_contribs[-len(normalized_contribs):] = normalized_contribs
                token_contrib[layer_name] = padded_contribs
            processed_contributions.append(token_contrib)
        
        with open('./activations.json', 'w') as f:
            json.dump({
                'tokens': self.all_tokens,
                'activations': self.activations_per_token,
                'contributions': processed_contributions
            }, f)
        
        self.previous_activations = {}
@torch.no_grad()
def generate(args, model, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None, sampC=10):
    """
    Generate tokens and track activations, including processing the initial prompt.
    Now includes tracking of token contributions.
    """
    if sampC == 1:
        tracker = ActivationTracker(model)
        tracker.add_hooks()

    rnns = ['rnn', 'ogrnn', 'rru', 'gru', 'moglstm', 'mingru', 'oggru']
    lstms = ['lstm']
    block_size = 1 if args.type in [rnns, lstms] else model.get_block_size()
    hiddens = None
    cells = None

    # Store previous hidden states for RNN/LSTM contribution tracking
    if args.type in rnns + lstms:
        for name, module in model.named_modules():
            if isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)):
                module.prev_hidden = None

    if sampC == 1:
        # Process the prompt tokens to track their activations and contributions
        for i in range(idx.size(1)):
            # Get the current token
            current_token = idx[:, i:i+1]

            # Forward pass for the current token
            if args.type in rnns:
                # Store previous hidden state for contribution calculation
                logits, hiddens, all_hiddens = model(current_token, return_prev_hiddens=True, prev_hiddens=hiddens)
                # Store the hidden state for the next iteration
                for name, module in model.named_modules():
                    if isinstance(module, nn.RNN):
                        module.prev_hidden = hiddens
            elif args.type in lstms:
                logits, hiddens, cells, all_hiddens = model(current_token, return_prev_hiddens=True, prev_hiddens=hiddens, prev_cells=cells)
                # Store the hidden state for the next iteration
                for name, module in model.named_modules():
                    if isinstance(module, nn.LSTM):
                        module.prev_hidden = (hiddens, cells)
            else:
                # For transformer-like models, the contribution calculation happens in the hooks
                logits, _ = model(current_token)

            # Scale logits by temperature
            logits = logits[:, -1, :] / temperature

            # Decode and track the token
            if sampC == 1:
                new_token = train_dataset.decode([current_token.item()])
                tracker.all_tokens.append(new_token)
                tracker.current_token_idx += 1

    # Generation loop
    for _ in range(max_new_tokens):
        # Crop context if necessary
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]

        # Forward pass with contribution tracking
        if args.type in rnns:
            # Store previous hidden state for contribution calculation
            logits, hiddens, all_hiddens = model(idx_cond, return_prev_hiddens=True, prev_hiddens=hiddens)
            # Update prev_hidden for contribution tracking
            for name, module in model.named_modules():
                if isinstance(module, nn.RNN):
                    module.prev_hidden = hiddens
        elif args.type in lstms:
            logits, hiddens, cells, all_hiddens = model(idx_cond, return_prev_hiddens=True, prev_hiddens=hiddens, prev_cells=cells)
            # Update prev_hidden for contribution tracking
            for name, module in model.named_modules():
                if isinstance(module, nn.LSTM):
                    module.prev_hidden = (hiddens, cells)
        else:
            logits, _ = model(idx_cond)

        # Scale logits by temperature
        logits = logits[:, -1, :] / temperature

        # Apply top_k filtering if specified
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)

        # Sample or take the most likely token
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)

        # Check for 0 tensor and stop generation if encountered
        if sampC == 1 and args.line_delim:
            if idx_next.item() == 0:
                break

        # Decode and track the new token
        if sampC == 1:
            new_token = train_dataset.decode([idx_next.item()])
            tracker.all_tokens.append(new_token)
            tracker.current_token_idx += 1

        # Update the input sequence
        idx = torch.cat((idx, idx_next), dim=1)

    if sampC == 1:
        tracker.save_activations('activation_data')
        tracker.remove_hooks()
        
        # Clean up prev_hidden attributes
        if args.type in rnns + lstms:
            for name, module in model.named_modules():
                if hasattr(module, 'prev_hidden'):
                    delattr(module, 'prev_hidden')

    return idx





def print_samples(args, num=10, prompt=None, do_sample=True, steper=100, temperature=1.0):
    """ samples from the model and pretty prints the decoded samples """
    if prompt is None:
        X_init = torch.zeros(num, 1, dtype=torch.long).to(args.device)
    else:
        prompt_encoded = train_dataset.encode(prompt)
        prompt_tensor = torch.tensor(prompt_encoded, dtype=torch.long).unsqueeze(0)
        zero_tensor = torch.zeros(1, dtype=torch.long).unsqueeze(0)
        X_init = torch.cat([zero_tensor, prompt_tensor], dim=1).repeat(num, 1).to(args.device)  # prompt_tensor.repeat(num, 1).to(args.device)

    top_k = args.top_k if args.top_k != -1 else None
    steps = train_dataset.get_output_length() - len(X_init[0]) if args.line_delim else steper
    X_samp = generate(args, model, X_init, steps, top_k=top_k, do_sample=do_sample, temperature=temperature, sampC=num).to(args.device)
    train_samples, test_samples, new_samples = [], [], []
    for i in range(X_samp.size(0)):
        row = X_samp[i, len(X_init[0]):].tolist()
        if args.line_delim:
            crop_index = row.index(0) if 0 in row else len(row)
            row = row[:crop_index]
        word_samp = train_dataset.decode(row)
        if train_dataset.contains(word_samp):
            train_samples.append(word_samp)
        elif test_dataset.contains(word_samp):
            test_samples.append(word_samp)
        else:
            new_samples.append(word_samp)
    print('-' * 80)
    for lst, desc in [(train_samples, 'in train'), (test_samples, 'in test'), (new_samples, 'new')]:
        print(f"{len(lst)} samples that are {desc}:")
        for word in lst:
            print('v' * 20)
            if prompt is not None:
                print(f"\033[1m{prompt}\033[0m", end="")
            print(word)
            print('^' * 20)
    print('-' * 80)

@torch.inference_mode()
def evaluate(model, dataset, batch_size=50, max_batches=None):
    model.eval()
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        batch = [t.to(args.device) for t in batch]
        X, Y = batch
        logits, loss = model(X, Y)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()
    model.train()  # reset model back to training mode
    return mean_loss

# -----------------------------------------------------------------------------
# helper functions for creating the training and test Datasets that emit words

import torch
from torch.utils.data import Dataset, DataLoader
import itertools

import torch
from torch.utils.data import Dataset, DataLoader
import os

import os
import torch
from torch.utils.data import Dataset

class StreamingCharDataset(Dataset):
    def __init__(self, file_path, chars, seq_len, offset=0, size=None):
        self.file_path = file_path
        self.chars = chars
        self.seq_len = seq_len
        self.stoi = {ch: i + 1 for i, ch in enumerate(chars)}
        self.itos = {i: s for s, i in self.stoi.items()}
        
        # Read the entire file into memory
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = f.read()
        
        # Calculate the total number of characters
        self.total_chars = len(self.data)
        self.offset = offset
        # Adjust length calculation to prevent accessing beyond data length
        max_length = max(0, self.total_chars - seq_len - offset)
        self._length = min(size, max_length) if size is not None else max_length
        
        # Add line_delim flag
        self.line_delim = False  # Default to streaming mode
    
    def __len__(self):
        return self._length
    
    def get_vocab_size(self):
        return len(self.chars) + 1
    
    def get_output_length(self):
        return self.seq_len
    
    def encode(self, sequence):
        return torch.tensor([self.stoi.get(w, 0) for w in sequence], dtype=torch.long)
    
    def decode(self, ix):
        return ''.join([self.itos.get(i, '') for i in ix if i in self.itos])

    def contains(self, sequence, chunk_size=1024*1024):
        """Check if sequence exists in dataset by iterating through self.data in chunks"""
        if not sequence or len(sequence) > self.seq_len:
            return False
        
        seq_len = len(sequence)
        buffer = ''
        start_pos = self.offset
        end_pos = self.offset + self._length if self._length else self.total_chars
        
        # Iterate through the data in chunks of characters
        for pos in range(start_pos, end_pos, chunk_size):
            chunk = self.data[pos: pos + chunk_size]
            buffer += chunk
            
            # Search for the sequence in the current buffer
            if sequence in buffer:
                return True
            
            # Keep the last (seq_len - 1) characters to handle overlapping
            if len(buffer) > seq_len:
                buffer = buffer[-(seq_len - 1):]
        
        return False

    def get_random_sequence(self, length=None):
        """Get a random sequence from the dataset"""
        if length is None:
            length = self.seq_len
        
        if length > self._length:
            return None
        
        start_idx = random.randint(0, self._length - length)
        return self.data[start_idx + self.offset: start_idx + self.offset + length]

    def get_sequence_at(self, idx, length=None):
        """Get a sequence at a specific index"""
        if length is None:
            length = self.seq_len
            
        if idx + length > self._length:
            return None
            
        return self.data[idx + self.offset: idx + self.offset + length]
    
    def __getitem__(self, idx):
        # Calculate character position
        pos = idx + self.offset
        sequence = self.data[pos: pos + self.seq_len + 1]  # Get seq_len + 1 characters for input/target
        
        if self.line_delim:
            # Original behavior with padding for line_delim mode
            if len(sequence) < self.seq_len + 1:
                sequence = sequence.ljust(self.seq_len + 1)
            ix = self.encode(sequence)
            x = torch.zeros(self.seq_len + 1, dtype=torch.long)
            y = torch.zeros(self.seq_len + 1, dtype=torch.long)
            x[1:1 + len(ix)] = ix[:-1]  # Input includes leading zero
            y[:len(ix)] = ix  # Target includes trailing zero
            y[len(ix) + 1:] = -1
        else:
            # New behavior for streaming mode: no zeros at all
            if len(sequence) < self.seq_len + 1:
                # Safety check: if sequence is empty or too short, pad with a default character
                pad_char = self.chars[0] if self.chars else ' '  # Use first char from vocab or space as fallback
                if sequence:  # If sequence has at least one character, use the last one
                    pad_char = sequence[-1]
                sequence = sequence + pad_char * (self.seq_len + 1 - len(sequence))
            
            # Encode the sequence
            ix = self.encode(sequence)
            
            # For streaming mode, x is the first seq_len characters and y is shifted by 1
            x = ix[:self.seq_len]  # First seq_len characters
            y = ix[1:self.seq_len + 1]  # Next seq_len characters
            
        return x, y

def get_unique_chars(file_path, chunk_size=1024*1024):
    """Get unique characters from file efficiently"""
    unique_chars = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            unique_chars.update(set(chunk))
    return sorted(list(unique_chars))

def create_datasets(input_file, line_delim=True, seq_len=100, seq2seq=False, input_col=None, output_col=None):
    """
    Create training and test datasets based on the input type.
    
    Args:
        input_file: Path to input file
        line_delim: If True, process as line-delimited data
        seq_len: Maximum sequence length
        seq2seq: If True, process as seq2seq data
        input_col: Input column name for seq2seq mode
        output_col: Output column name for seq2seq mode
    """
    if line_delim:
        # Existing line-delimited behavior
        with open(input_file, 'r', encoding='utf-8') as f:
            sequences = f.read().splitlines()
        sequences = [s.strip() for s in sequences]
        sequences = [s for s in sequences if s]
        
        unique_chars = sorted(list(set(''.join(sequences))))
        max_seq_length = max(len(seq) for seq in sequences)
        
        test_set_size = min(1000, int(len(sequences) * 0.1))
        rp = torch.randperm(len(sequences)).tolist()
        train_sequences = [sequences[i] for i in rp[:-test_set_size]]
        test_sequences = [sequences[i] for i in rp[-test_set_size:]]
        
        train_dataset = CharDataset(train_sequences, unique_chars, max_seq_length)
        test_dataset = CharDataset(test_sequences, unique_chars, max_seq_length)
        
    else:
        unique_chars = get_unique_chars(input_file)
        
        # Read the file to determine the number of characters
        with open(input_file, 'r', encoding='utf-8') as f:
            data = f.read()
        
        total_chars = len(data)
        total_sequences = max(0, total_chars - seq_len - 1)
        
        test_size = min(10000, int(total_sequences * 0.1))
        train_size = total_sequences - test_size
        
        # Initialize datasets with correct sizes
        train_dataset = StreamingCharDataset(input_file, unique_chars, seq_len, 0, train_size)
        train_dataset.line_delim = False
        test_dataset = StreamingCharDataset(input_file, unique_chars, seq_len, train_size, test_size)
        test_dataset.line_delim = False
    
    # Print dataset information
    if seq2seq:
        input_vocab_size, output_vocab_size = train_dataset.get_vocab_size()
        print(f"number of examples in the dataset: {len(train_dataset) + len(test_dataset)}")
        print(f"max sequence length: {seq_len}")
        print(f"number of unique characters in input vocabulary: {input_vocab_size}")
        print(f"number of unique characters in output vocabulary: {output_vocab_size}")
        print("input vocabulary:", ''.join(train_dataset.chars_input))
        print("output vocabulary:", ''.join(train_dataset.chars_output))
    else:
        print(f"number of examples in the dataset: {len(train_dataset) + len(test_dataset)}")
        print(f"max sequence length: {seq_len}")
        print(f"number of unique characters in the vocabulary: {len(unique_chars)}")
        print("vocabulary:", ''.join(unique_chars))
        
    print(f"split up the dataset into {len(train_dataset)} training examples and {len(test_dataset)} test examples")
    
    return train_dataset, test_dataset

class CharDataset(Dataset):
    """Original CharDataset for line_delim mode"""
    def __init__(self, data, chars, max_seq_length):
        self.data = data
        self.chars = chars
        self.max_seq_length = max_seq_length
        self.stoi = {ch: i + 1 for i, ch in enumerate(chars)}
        self.itos = {i: s for s, i in self.stoi.items()}

    def __len__(self):
        return len(self.data)

    def contains(self, sequence):
        return sequence in self.data

    def get_vocab_size(self):
        return len(self.chars) + 1

    def get_output_length(self):
        return self.max_seq_length + 1

    def encode(self, sequence):
        ix = torch.tensor([self.stoi.get(w, 0) for w in sequence], dtype=torch.long)
        return ix

    def decode(self, ix):
        word = ''.join([self.itos.get(i, '') for i in ix if i in self.itos])
        return word

    def __getitem__(self, idx):
        sequence = self.data[idx]
        ix = self.encode(sequence)
        x = torch.zeros(self.max_seq_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_seq_length + 1, dtype=torch.long)
        x[1:1 + len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix) + 1:] = -1
        return x, y

class InfiniteDataLoader:
    def __init__(self, dataset, **kwargs):
        train_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)
        
    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        return batch

def initialize_model(args, config):
    if args.type == 'transformer':
        model = Transformer(config)
    elif args.type == 'gpt2':
        model = OGTransformer(config)
    elif args.type == 'bigram':
        model = Bigram(config)
    elif args.type == 'mlp':
        model = MLP(config)
    elif args.type == 'mlp_gelu':
        model = MLPOG_GELU(config)
    elif args.type == 'mlp_lm':
        model = MLPLLM(config)
    elif args.type == 'mlp_pca':
        model = ComplexMLP(config)
    elif args.type == 'mlp_og':
        model = MLPOG(config)
    elif args.type == 'mlp_adalin':
        model = ADALIN(config)
    elif args.type == 'rnn':
        model = RNN(config, cell_type='rnn', training=False)
    elif args.type == 'ogrnn':
        model = RNN(config, cell_type='ogrnn', training=False)
    elif args.type == 'rru':
        model = RNN(config, cell_type='rru', training=False)
    elif args.type == 'gru':
        model = RNN(config, cell_type='gru', training=False)
    elif args.type == 'moglstm':
        model = RNN(config, cell_type='moglstm', training=False)
    elif args.type == 'oggru':
        model = RNN(config, cell_type='oggru', training=False)
    elif args.type == 'lstm':
        model = RNN(config, cell_type='lstm', training=False)
    elif args.type == 'bow':
        model = BoW(config)
    elif args.type == 'rwkv':
        model = RWKV5(config)
    elif args.type == 'mingru':
        model = minGRULM(num_tokens=config.vocab_size, dim=args.n_embd, depth=args.n_layer, blocks=config.block_size)
    elif args.type == 'convnet':
        model = CharConvNet(config)
    elif args.type == 'hypermixer':
        model = Mamba(config)
    else:
        raise ValueError(f"Unknown model type: {args.type}")
    return model

import torch
from torch.optim import Optimizer

# Assuming all custom optimizer classes (AdamZ, CaAdam, WarpAdam, ADOPT, NSGDA, BGEAdam, SNRAdam, etc.)
# are defined/imported in the same module or appropriately imported.

def initialize_optimizer(
    model, 
    args, 
    alpha=0.9, 
    beta1=0.9, 
    beta2=0.999, 
    momentum=0.9, 
    hypergrad_lr=1e-8, 
    grokfast_alpha=0.98, 
    cautious_factor=0.0, 
    initial_accumulator_value=0.0, 
    lr_decay=0.0, 
    delta=0.1, 
    eps=1e-16,
    overshoot_factor=0.5,              # For AdamZ
    stagnation_factor=1.2,              # For AdamZ
    stagnation_threshold=0.2,           # For AdamZ
    patience=100,                        # For AdamZ
    stagnation_period=10,                # For AdamZ
    max_grad_norm=1.0,                   # For AdamZ
    lr_bounds=(1e-7, 1.0),               # For AdamZ
    scaling_method='multiplicative',     # For CaAdam
    gamma=0.95,                          # For CaAdam
    clip_value=0.1,                      # For ADOPT
    layer_wise=True,                     # For NSGDA
    entropy_weight=0.01,                 # For BGEAdam
    beta1_max=0.9, beta1_min=0.5,        # For BGEAdam
    beta2_max=0.999, beta2_min=0.9       # For BGEAdam
):
    if args.optim == "lamb":
        optimizer = Lamb(
            model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay, 
            betas=(beta1, beta2)
        )
    elif args.optim == "lambhd":
        optimizer = LambHD(
            model.parameters(), 
            lr=0.0, 
            betas=(beta1, beta2), 
            weight_decay=args.weight_decay, 
            lr_lower_bound=0.0, 
            lr_upper_bound=args.learning_rate, 
            hypergrad_lr=hypergrad_lr
        )
    elif args.optim == "adamhd":
        optimizer = AdamHD(
            model.parameters(), 
            lr=0.0, 
            betas=(beta1, beta2), 
            weight_decay=args.weight_decay, 
            hypergrad_lr=hypergrad_lr
        )
    elif args.optim == "gfadamw":
        optimizer = GrokFastAdamW(
            model.parameters(), 
            lr=args.learning_rate, 
            betas=(beta1, beta2), 
            grokfast_alpha=grokfast_alpha
        )
    elif args.optim == "gflamb":
        optimizer = GrokFastLamb(
            model.parameters(), 
            lr=args.learning_rate, 
            betas=(beta1, beta2), 
            grokfast_alpha=grokfast_alpha
        )
    elif args.optim == "gflambhd":
        optimizer = AdamAtan2(
            model.parameters(), 
            lr=args.learning_rate, 
            betas=(beta1, beta2), 
            cautious_factor=cautious_factor
        )
    elif args.optim == "adamz":
        optimizer = AdamZ(
            model.parameters(),
            lr=args.learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=args.weight_decay,
            amsgrad=False,  # Modify as needed or pass via args
            overshoot_factor=overshoot_factor,
            stagnation_factor=stagnation_factor,
            stagnation_threshold=stagnation_threshold,
            patience=patience,
            stagnation_period=stagnation_period,
            max_grad_norm=max_grad_norm,
            lr_bounds=lr_bounds
        )
    elif args.optim == "caadam":
        optimizer = CaAdam(
            params=model.parameters(),
            model=model,
            lr=args.learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=args.weight_decay,
            amsgrad=False,  # Modify as needed or pass via args
            scaling_method=scaling_method,
            gamma=gamma
        )
    elif args.optim == "warpadam":
        optimizer = WarpAdam(
            params=model.parameters(),
            model=model,
            lr=args.learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=args.weight_decay,
            amsgrad=False  # Modify as needed or pass via args
        )
    elif args.optim == "adopt":
        optimizer = ADOPT(
            params=model.parameters(),
            lr=args.learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=args.weight_decay,
            amsgrad=False,  # Modify as needed or pass via args
            clip_value=clip_value
        )
    elif args.optim == "nsgda":
        optimizer = NSGDA(
            params=model.parameters(),
            lr=args.learning_rate,
            momentum=momentum,
            layer_wise=layer_wise,
            eps=eps
        )
    elif args.optim == "bgeadam":
        optimizer = BGEAdam(
            params=model.parameters(),
            lr=args.learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=args.weight_decay,
            amsgrad=False,  # Modify as needed or pass via args
            alpha=alpha,
            entropy_weight=entropy_weight,
            beta1_max=beta1_max,
            beta1_min=beta1_min,
            beta2_max=beta2_max,
            beta2_min=beta2_min
        )
    elif args.optim == "snradam":
        optimizer = SNRAdam(
            params=model.parameters(),
            lr=args.learning_rate,
            betas=(beta1, beta2),
            weight_decay=args.weight_decay,
            eps=eps
        )
    elif args.optim == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.learning_rate, 
            betas=(beta1, beta2)
        )
    elif args.optim == "adamax":
        optimizer = torch.optim.Adamax(
            model.parameters(), 
            lr=args.learning_rate, 
            betas=(beta1, beta2)
        )
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=args.learning_rate * 24
        )
    elif args.optim == "sgdmomentum":
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=args.learning_rate * 24, 
            momentum=momentum
        )
    elif args.optim == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(), 
            lr=args.learning_rate, 
            alpha=alpha, 
            momentum=momentum
        )
    elif args.optim == "adagrad":
        optimizer = torch.optim.Adagrad(
            model.parameters(), 
            lr=args.learning_rate * 24, 
            initial_accumulator_value=initial_accumulator_value, 
            lr_decay=lr_decay
        )
    elif args.optim == "adadelta":
        optimizer = torch.optim.Adagrad(
            model.parameters(), 
            lr=args.learning_rate * 100,
        )
    elif args.optim == "adamp":
        optimizer = AdamP(
            model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay, 
            betas=(beta1, beta2), 
            delta=delta
        )
    elif args.optim == "adabelief":
        optimizer = AdaBelief(
            model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay, 
            betas=(beta1, beta2), 
            eps=eps, 
            print_change_log=False
        )
    else:
        raise ValueError(f"Optimizer '{args.optim}' is not supported.")
    
    return optimizer


from tqdm import tqdm

def train_model(model, optimizer, train_dataset, test_dataset, args, max_steps):
    model.train()
    batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
    best_val_loss = None
    grads = None
    step = 0

    # Initialize tqdm progress bar
    with tqdm(total=max_steps, desc="Training Progress") as pbar:
        while step < max_steps:
            t0 = time.time()
            # Get the next batch
            batch = batch_loader.next()
            batch = [t.to(args.device) for t in batch]
            X, Y = batch
            # Forward pass
            logits, loss = model(X, Y)
            # Backward pass and optimization
            model.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            # Update tqdm bar
            pbar.update(1)
            pbar.set_postfix(loss=loss.item())

            # Validation
            if step % 10 == 0:
                val_loss = evaluate(model, test_dataset, batch_size=10, max_batches=10)
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    pbar.set_postfix(best_val_loss=best_val_loss)
            step += 1

    return best_val_loss

from tqdm import tqdm
import optuna
from optuna.trial import TrialState
from optuna import Trial
from optuna.exceptions import TrialPruned

import numpy as np
from statistics import mean, stdev
from tqdm import tqdm

def objective(
    trial: Trial,
    args,
    train_dataset,
    test_dataset,
    embed,
    layer,
    head,
    model_type,
    n_startup_trials,
    train_only
):
    try:
        study = trial.study
        completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
        n_completed = len(completed_trials)

        # Determine possible models based on the type
        if model_type == 'automlp':
            models = ['mlp_og', 'mlp_lm', 'mlp', 'mlp_gelu', 'mlp_pca', 'bow', 'mlp_adalin']
        elif model_type == 'autornn':
            models = ['rnn', 'ogrnn', 'rru', 'gru', 'moglstm', 'lstm', 'oggru']
        elif model_type == 'transformerlikes':
            models = ['transformer', 'gpt2', 'rwkv', 'convnet', 'hypermixer']
        elif model_type == 'autoall':
            models = [
                'mlp_og', 'mlp_lm', 'mlp', 'mlp_gelu', 'mlp_pca', 'convnet',
                'bow', 'bigram', 'mingru',
                'transformer', 'gpt2', 'rwkv', 'mlp_adalin', 'hypermixer'
            ]
        elif model_type == 'autoallnomingru':
            models = [
                'mlp_og', 'mlp_lm', 'mlp', 'mlp_gelu', 'mlp_pca', 'convnet',
                'bow', 'bigram',
                'transformer', 'gpt2', 'rwkv', 'mlp_adalin', 'hypermixer'
            ]
        else:
            models = [model_type]

        # Suggest model type first
        args.type = trial.suggest_categorical('model', models)

        # Define model-specific learning rate ranges
        learning_rate_ranges = {
            'mlp_og': (1e-4, 1e-2),
            'mlp_lm': (5e-5, 1e-2),
            'mlp_adalin': (5e-5, 1e-3),
            'mlp': (5e-5, 1e-2),
            'mlp_gelu': (1e-5, 1e-2),
            'mlp_pca': (1e-5, 1e-2),
            'bow': (5e-5, 1e-2),
            'rnn': (5e-5, 1e-2),
            'ogrnn': (1e-5, 1e-2),
            'rru': (5e-5, 1e-2),
            'gru': (1e-5, 1e-2),
            'oggru': (1e-5, 1e-2),
            'lstm': (1e-5, 1e-2),
            'moglstm': (5e-5, 1e-2),
            'bigram': (1e-1, 1e-10),
            'mingru': (5e-5, 1e-2),
            'transformer': (5e-5, 5e-3),
            'gpt2': (5e-5, 5e-3),
            'rwkv': (1e-5, 5e-3),
            'convnet': (5e-5, 1e-2),
        }

        # Suggest learning rate based on the selected model
        lr_min, lr_max = learning_rate_ranges.get(args.type, (1e-5, 1.0))
        suggested_lr = trial.suggest_float('learning_rate', lr_min, lr_max, log=True)

        # Suggest embedding size
        if args.optimi == 'model':
            min_embd = 1
            min_layer = 2 if args.type == 'rwkv' else 1
        else:
            min_embd = embed
            min_layer = layer

        if args.type in ['transformer', 'gpt2', 'rwkv']:
            embed_size = embed
        else:
            embed_size = trial.suggest_int('embed_size', min_embd, embed)
            # Ensure embed_size is divisible by args.n_head
            while embed_size % args.n_head != 0:
                embed_size = trial.suggest_int('embed_size', min_embd, embed)

        # Suggest number of layers
        layer_count = trial.suggest_int('layer_count', min_layer, layer)

        # Suggest number of heads if the model is transformer-like
        if args.type in ['transformer', 'gpt2', 'rwkv']:
            if args.optimi == 'model':
                valid_head_counts = [
                    h for h in range(2, head + 1) if embed_size % h == 0
                ]
                if not valid_head_counts:
                    valid_head_counts = [1]  # Fallback to 1 if no valid heads
                n_heads = trial.suggest_categorical('n_heads', valid_head_counts)
            else:
                n_heads = trial.suggest_int('n_heads', head, head)
        else:
            n_heads = 1  # Default to 1 if not transformer-like

        # Suggest optimizer hyperparameters
        # Initialize with default values
        momentum = 0.9
        beta1 = 0.9
        beta2 = 0.999
        hypergrad_lr = 1e-8
        alpha = 0.9
        initial_accumulator_value = 0.0
        lr_decay = 0.0
        grokfast_alpha = 0.0
        cautious_factor = 0.0
        delta = 0.0
        eps = 0.0
        scaling_method = 'multiplicative'
        gamma = 0.95
        clip_value = 0.1
        layer_wise = True
        entropy_weight = 0.01
        beta1_max = 0.9
        beta1_min = 0.5
        beta2_max = 0.999
        beta2_min = 0.9
        overshoot_factor = 0.5
        stagnation_factor = 1.2
        stagnation_threshold = 0.2
        patience = 100
        stagnation_period = 10
        max_grad_norm = 1.0
        lr_min_bound = 1e-7
        lr_max_bound = 1.0

        # Suggest hyperparameters based on the selected optimizer
        if args.optim in ["sgdmomentum", "rmsprop"]:
            momentum = trial.suggest_float('momentum', 0.5, 1.0)
        else:
            momentum = 0.9  # Default value

        if args.optim in [
            "adam", "lamb", "adamhd", "lambhd", "gflambhd",
            "gflamb", "adamp", "adabelief", "AdamZ", "CaAdam",
            "WarpAdam", "ADOPT", "BGEAdam", "SNRAdam"
        ]:
            beta1 = trial.suggest_float('beta1', 0.0, 0.99999)
            beta2 = trial.suggest_float('beta2', 0.9, 0.99999)
        else:
            beta1 = 0.9  # Default value
            beta2 = 0.999  # Default value

        if args.optim in ["adamhd", "lambhd"]:
            hypergrad_lr = trial.suggest_float('hyper_lr', 1e-12, 1e-4, log=True)
        else:
            hypergrad_lr = 1e-8  # Default value

        if args.optim == "rmsprop":
            alpha = trial.suggest_float('alpha', 0.9, 0.9999)
        else:
            alpha = 0.9  # Default value

        if args.optim == "adagrad":
            initial_accumulator_value = trial.suggest_float('init_accum_val', 1e-4, 1.0)
            lr_decay = trial.suggest_float('lrdecay', 0.0, 1.0)
        else:
            initial_accumulator_value = 0.0  # Default value
            lr_decay = 0.0  # Default value

        if args.optim in ["gflamb", "gfadamw"]:
            grokfast_alpha = trial.suggest_float('gf_alpha', 0.5, 1.0)
        else:
            grokfast_alpha = 0.0  # Default value

        if args.optim == "gflambhd":
            cautious_factor = trial.suggest_float('caut_fact', 1e-18, 1.0)
        else:
            cautious_factor = 0.0  # Default value

        if args.optim == "adamp":
            delta = trial.suggest_float('delta', 0.0, 1.0)
        else:
            delta = 0.0  # Default value

        if args.optim == "adabelief":
            eps = trial.suggest_float('eps', 1e-16, 1e-8)
        else:
            eps = 1e-16  # Default value

        # Suggest hyperparameters for newly added optimizers
        if args.optim == "adamz":
            overshoot_factor = trial.suggest_float('overshoot_factor', 0.1, 1.0, log=False)
            stagnation_factor = trial.suggest_float('stagnation_factor', 1.0, 2.0, log=False)
            stagnation_threshold = trial.suggest_float('stagnation_threshold', 0.0, 1.0, log=False)
            patience = trial.suggest_int('patience', 50, 200)
            stagnation_period = trial.suggest_int('stagnation_period', 5, 20)
            max_grad_norm = trial.suggest_float('max_grad_norm', 0.5, 2.0, log=False)
            lr_min_bound = trial.suggest_float('lr_min_bound', 1e-7, 1e-6, log=True)
            lr_max_bound = trial.suggest_float('lr_max_bound', 1e-3, 1e-1, log=True)
            lr_bounds = (lr_min_bound, lr_max_bound)
        else:
            overshoot_factor = 0.5
            stagnation_factor = 1.2
            stagnation_threshold = 0.2
            patience = 100
            stagnation_period = 10
            max_grad_norm = 1.0
            lr_bounds = (1e-7, 1.0)

        if args.optim == "caadam":
            scaling_method = trial.suggest_categorical(
                'scaling_method', ['additive', 'multiplicative', 'depth']
            )
            gamma = trial.suggest_float('gamma', 0.8, 1.0, log=False)
        else:
            scaling_method = 'multiplicative'
            gamma = 0.95

        if args.optim == "adopt":
            clip_value = trial.suggest_float('clip_value', 0.0, 1.0, log=False)
        else:
            clip_value = 0.1  # Default value

        if args.optim == "nsgda":
            layer_wise = trial.suggest_categorical('layer_wise', [True, False])
        else:
            layer_wise = True  # Default value

        if args.optim == "bgeadam":
            entropy_weight = trial.suggest_float('entropy_weight', 0.0, 0.1, log=False)
            beta1_max = trial.suggest_float('beta1_max', 0.8, 0.99, log=False)
            beta1_min = trial.suggest_float('beta1_min', 0.3, 0.8, log=False)
            beta2_max = trial.suggest_float('beta2_max', 0.99, 0.9999, log=False)
            beta2_min = trial.suggest_float('beta2_min', 0.8, 0.99, log=False)
        else:
            entropy_weight = 0.01
            beta1_max = 0.9
            beta1_min = 0.5
            beta2_max = 0.999
            beta2_min = 0.9

        # Update the args with the suggested hyperparameters
        args.learning_rate = suggested_lr
        args.n_embd = embed_size
        args.n_layer = layer_count
        args.n_head = n_heads

        # Update args with optimizer-specific hyperparameters
        args.momentum = momentum
        args.beta1 = beta1
        args.beta2 = beta2
        args.hypergrad_lr = hypergrad_lr
        args.alpha = alpha
        args.initial_accumulator_value = initial_accumulator_value
        args.lr_decay = lr_decay
        args.grokfast_alpha = grokfast_alpha
        args.cautious_factor = cautious_factor
        args.delta = delta
        args.eps = eps
        args.scaling_method = scaling_method
        args.gamma = gamma
        args.clip_value = clip_value
        args.layer_wise = layer_wise
        args.entropy_weight = entropy_weight
        args.beta1_max = beta1_max
        args.beta1_min = beta1_min
        args.beta2_max = beta2_max
        args.beta2_min = beta2_min
        args.overshoot_factor = overshoot_factor
        args.stagnation_factor = stagnation_factor
        args.stagnation_threshold = stagnation_threshold
        args.patience = patience
        args.stagnation_period = stagnation_period
        args.max_grad_norm = max_grad_norm
        args.lr_bounds = lr_bounds

        # Reinitialize the model and optimizer
        config = ModelConfig(
            vocab_size=train_dataset.get_vocab_size(),
            block_size=train_dataset.get_output_length(),
            n_layer=layer_count,
            n_head=n_heads,
            n_embd=embed_size,
            n_embd2=args.n_embd2  # Retain any other settings
        )
        temp_model = initialize_model(args, config)
        temp_model.to(args.device)

        # Initialize the optimizer with the suggested hyperparameters
        temp_optimizer = initialize_optimizer(
            temp_model,
            args,
            alpha=alpha,
            beta1=beta1,
            beta2=beta2,
            momentum=momentum,
            hypergrad_lr=hypergrad_lr,
            grokfast_alpha=grokfast_alpha,
            cautious_factor=cautious_factor,
            initial_accumulator_value=initial_accumulator_value,
            lr_decay=lr_decay,
            delta=delta,
            eps=eps,
            overshoot_factor=overshoot_factor,
            stagnation_factor=stagnation_factor,
            stagnation_threshold=stagnation_threshold,
            patience=patience,
            stagnation_period=stagnation_period,
            max_grad_norm=max_grad_norm,
            lr_bounds=lr_bounds,
            scaling_method=scaling_method,
            gamma=gamma,
            clip_value=clip_value,
            layer_wise=layer_wise,
            entropy_weight=entropy_weight,
            beta1_max=beta1_max,
            beta1_min=beta1_min,
            beta2_max=beta2_max,
            beta2_min=beta2_min
        )

        # Training parameters
        eval_steps = 501  # Adjust as needed
        temp_batch_loader = InfiniteDataLoader(
            train_dataset,
            batch_size=args.batch_size,
            pin_memory=True,
            num_workers=0
        )

        temp_model.train()
        best_val_loss = None

        # Initialize lists to track training and validation losses
        training_losses = []
        validation_losses = []

        # Initialize TQDM progress bar
        progress_bar = tqdm(
            range(eval_steps),
            desc=f"Training {args.type}",
            leave=False
        )

        for step in progress_bar:
            # Get the next batch
            batch = temp_batch_loader.next()
            batch = [t.to(args.device) for t in batch]
            X, Y = batch

            # Forward pass
            logits, loss = temp_model(X, Y)

            # Record training loss
            training_losses.append(loss.item())

            # Backward pass and optimization
            temp_model.zero_grad(set_to_none=True)
            loss.backward()
            temp_optimizer.step()

            # Periodically evaluate
            if step % 5 == 0:
                if train_only:
                    # Use training loss for evaluation
                    current_train_loss = loss.item()
                    if math.isnan(current_train_loss):
                        raise TrialPruned("Pruning because training loss is NaN")
                    training_losses.append(current_train_loss)

                    if step > 10:
                        train_losses_array = np.array(training_losses)

                        # Calculate metrics for training loss behavior
                        loss_decrease = training_losses[0] - training_losses[-1]
                        loss_stability = -stdev(train_losses_array)  # Lower std dev is better
                        loss_trend = training_losses[-1] - training_losses[0]  # Negative means decreasing

                        # Define penalties and rewards
                        oscillation_penalty = stdev(train_losses_array)  # Higher std dev penalizes
                        upward_trend_penalty = max(0, loss_trend)  # Positive trend (increase) penalizes
                        steep_decrease_reward = loss_decrease  # Larger decrease rewards

                        # Combine into a single objective metric
                        combined_metric = current_train_loss + oscillation_penalty + upward_trend_penalty - steep_decrease_reward
                        trial.report(combined_metric, step)
                    else:
                        combined_metric = current_train_loss

                    # Update progress bar with training loss
                    progress_bar.set_postfix({'train_loss': combined_metric})

                else:
                    # Original behavior: evaluate on validation set
                    val_loss = evaluate(
                        temp_model,
                        test_dataset,
                        batch_size=10,
                        max_batches=10
                    )
                    if math.isnan(loss):
                        raise TrialPruned("Pruning because loss is NaN")
                    validation_losses.append(val_loss)
                    if step > 10:
                        validation_lossess = np.array(validation_losses)

                        # Calculate metrics for training loss behavior
                        loss_decrease = validation_losses[0] - validation_losses[-1]
                        loss_stability = -stdev(validation_lossess)  # Lower std dev is better
                        loss_trend = validation_losses[-1] - validation_losses[0]  # Negative means decreasing

                        # Define penalties and rewards
                        oscillation_penalty = stdev(validation_lossess)  # Higher std dev penalizes
                        upward_trend_penalty = max(0, loss_trend)  # Positive trend (increase) penalizes
                        steep_decrease_reward = loss_decrease  # Larger decrease rewards

                        # Combine into a single objective metric
                        combined_metric = val_loss + oscillation_penalty + upward_trend_penalty - steep_decrease_reward
                        trial.report(combined_metric, step)
                    else:
                        combined_metric = val_loss

                    # Update progress bar with validation loss
                    progress_bar.set_postfix({'val_loss': combined_metric})

                # Handle pruning
                if trial.should_prune() and n_completed > n_startup_trials:
                    raise TrialPruned()

                if not train_only:
                    if best_val_loss is None or val_loss < best_val_loss:
                        best_val_loss = val_loss
                else:
                    if best_val_loss is None or combined_metric < best_val_loss:
                        best_val_loss = combined_metric

        progress_bar.close()

        if train_only:
            training_losses = np.array(training_losses)

            # Calculate metrics for training loss behavior
            loss_decrease = training_losses[0] - training_losses[-1]
            loss_stability = -stdev(training_losses)  # Lower std dev is better
            loss_trend = training_losses[-1] - training_losses[0]  # Negative means decreasing

            # Define penalties and rewards
            oscillation_penalty = stdev(training_losses)  # Higher std dev penalizes
            upward_trend_penalty = max(0, loss_trend)  # Positive trend (increase) penalizes
            steep_decrease_reward = loss_decrease  # Larger decrease rewards

            # Combine into a single objective metric
            combined_metric = best_val_loss + oscillation_penalty + upward_trend_penalty - steep_decrease_reward

            # Alternatively, you can normalize and weigh each component
            # combined_metric = best_val_loss + 0.1 * oscillation_penalty + 0.1 * upward_trend_penalty - 0.2 * steep_decrease_reward

        else:
            validation_losses = np.array(validation_losses)

            # Calculate metrics for validation loss behavior
            loss_decrease = validation_losses[0] - validation_losses[-1]
            loss_stability = -stdev(validation_losses)  # Lower std dev is better
            loss_trend = validation_losses[-1] - validation_losses[0]  # Negative means decreasing

            # Define penalties and rewards
            oscillation_penalty = stdev(validation_losses)  # Higher std dev penalizes
            upward_trend_penalty = max(0, loss_trend)  # Positive trend (increase) penalizes
            steep_decrease_reward = loss_decrease  # Larger decrease rewards

            # Combine into a single objective metric
            combined_metric = best_val_loss + oscillation_penalty + upward_trend_penalty - steep_decrease_reward

            # Alternatively, you can normalize and weigh each component
            # combined_metric = best_val_loss + 0.1 * oscillation_penalty + 0.1 * upward_trend_penalty - 0.2 * steep_decrease_reward

        return combined_metric

    except TrialPruned:
        # Re-raise the TrialPruned exception to let Optuna handle pruning
        raise
    except Exception as e:
        if train_only:
            # Prune the trial in case of any exceptions during training evaluation
            raise TrialPruned()
        else:
            # Prune the trial in case of any exceptions during validation evaluation
            raise TrialPruned()
        # Alternatively, you can return a high loss value
        # return float('inf')
    except ValueError:
        raise TrialPruned()


# -----------------------------------------------------------------------------
import argparse
import torch
import os
import sys
import time
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import ast  # Added for safely evaluating best_params.txt
# Make sure to import other necessary modules and functions
# from your_project import ModelConfig, initialize_model, initialize_optimizer, evaluate, print_samples, create_datasets, InfiniteDataLoader

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # Parse command line args
    parser = argparse.ArgumentParser(description="Make More")
    # System/input/output
    parser.add_argument('--input-file', '-i', type=str, default='names.txt', help="input file with things one per line or normal text")
    parser.add_argument('--optim', type=str, default='gfadamw', help="Optimizer")
    parser.add_argument('--prompt', '-p', type=str, default='A', help="Prompt")
    parser.add_argument('--work-dir', '-o', type=str, default='out', help="output working directory")
    parser.add_argument('--resume', action='store_true', help="when this flag is used, we will resume optimization from existing model in the workdir")
    parser.add_argument('--optimize', action='store_true', help="when this flag is used, we will optimize hyperparameters")
    parser.add_argument('--sample-only', action='store_true', help="just sample from the model and quit, don't train")
    parser.add_argument('--num-workers', '-n', type=int, default=4, help="number of data workers for both train/test")
    parser.add_argument('--max-steps', type=int, default=-1, help="max number of optimization steps to run for, or -1 for infinite.")
    parser.add_argument('--num-samples', type=int, default=50, help="number of samples to generate")
    parser.add_argument('--device', type=str, default='cuda', help="device to use for compute, examples: cpu|cuda|cpu:2|mps")
    parser.add_argument('--seed', type=int, default=3407, help="seed")
    # Sampling
    parser.add_argument('--top-k', type=int, default=-1, help="top-k for sampling, -1 means no top-k")
    parser.add_argument('--stepscount', type=int, default=200, help="Number of optimization steps for Optuna trials")
    parser.add_argument('--do-sample', type=int, default=-1, help="Should we sample? 0 means ArgMax")
    # Model
    parser.add_argument('--type', type=str, default='transformer', help="model class type to use, bigram|mlp|rnn|gru|bow|transformer")
    parser.add_argument('--optimi', type=str, default='model', help="What should we optimize? model|optim")
    parser.add_argument('--autotype', type=str, default='all', help="model class to use, all|no-rnns|no-mlps|no-transformers|rnn-only|transformer-only|mlp-only")
    parser.add_argument('--n-layer', type=int, default=4, help="maximum number of layers")
    parser.add_argument('--n-head', type=int, default=4, help="maximum number of heads (in a transformer)")
    parser.add_argument('--n-embd', type=int, default=64, help="maximum number of feature channels in the model")
    parser.add_argument('--n-embd2', type=int, default=64, help="number of feature channels elsewhere in the model")
    # Optimization
    parser.add_argument('--batch-size', '-b', type=int, default=32, help="batch size during optimization")
    parser.add_argument('--learning-rate', '-l', type=float, default=5e-3, help="learning rate")
    parser.add_argument('--weight-decay', '-w', type=float, default=0.0, help="weight decay (Default: 0.0)")
    # New Arguments
    parser.add_argument('--seq_len', type=int, default=100, help="Sequence length for normal text files")
    parser.add_argument('--steps_sample', type=int, default=100, help="Sequence length for normal text files")
    parser.add_argument('--train_only', type=int, default=0, help="Sequence length for normal text files")
    parser.add_argument('--line_delim', action='store_true', help="Use line-delimited behavior. If not set, uses normal text processing.")
    args = parser.parse_args()

    args.n_embd2 = args.n_embd * 4
    train_only = args.train_only


    # System inits
    #torch.manual_seed(args.seed)
    os.makedirs(args.work_dir, exist_ok=True)

    # Init datasets
    train_dataset, test_dataset = create_datasets(args.input_file, line_delim=args.line_delim, seq_len=args.seq_len)
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_output_length()

    print(f"Dataset determined that: vocab_size={vocab_size}, block_size={block_size}")

    def save_optimizer_state(optimizer, path):
        torch.save(optimizer.state_dict(), path)

    # Load optimizer state
    def load_optimizer_state(optimizer, path):
        optimizer.load_state_dict(torch.load(path))

    # Function to optimize learning rate and hyperparameters
    def optimize_hyperparameters(args, train_dataset, test_dataset):
        """
        Optimizes hyperparameters using Optuna.
    
        Args:
            args: Configuration arguments.
            train_dataset: Training dataset.
            test_dataset: Testing dataset.
    
        Returns:
            dict: Best hyperparameters found by Optuna.
        """
        # Initialize the sampler with a seed and number of startup trials
        n_startup_trials = 10
        sampler = TPESampler(seed=args.seed, n_startup_trials=n_startup_trials, n_ei_candidates=100)
    
        # Initialize the HyperbandPruner for early stopping of unpromising trials
        pruner = HyperbandPruner()
    
        # Create the Optuna study with the specified direction, sampler, and pruner
        study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
    
        # Extract necessary parameters from args
        embed = args.n_embd
        layer = args.n_layer
        head = args.n_head
        model_type = args.type
        train_only = args.train_only  # Ensure this attribute exists in args
    
        # Run the optimization process
        study.optimize(
            lambda trial: objective(
                trial,
                args,
                train_dataset,
                test_dataset,
                embed,
                layer,
                head,
                model_type,
                n_startup_trials,
                train_only
            ),
            n_trials=n_startup_trials * 5,  # Number of trials (adjust as needed)
            timeout=600000,  # Timeout in seconds (adjust as needed)
            n_jobs=1  # Number of parallel jobs (set >1 for parallel optimization)
        )
    
        # Retrieve the best hyperparameters found
        best_params = study.best_params
    
        print(f"Best parameters found: {best_params}")
        return best_params

    best_params = None

    # Optimize the learning rate and model hyperparameters or load from resume
    if args.optimize:
        best_params = optimize_hyperparameters(args, train_dataset, test_dataset)
    elif args.resume:
        best_params_path = os.path.join(args.work_dir, 'best_params.txt')
        if os.path.exists(best_params_path):
            with open(best_params_path, 'r') as f:
                best_params = ast.literal_eval(f.read())
            print(f"Loaded best_params from {best_params_path}")
        else:
            print(f"Error: {best_params_path} does not exist. Cannot resume without best_params.")
            sys.exit(1)

    if best_params:
        # Set the best parameters in args
        args.learning_rate = best_params.get('learning_rate', args.learning_rate)
        
        if args.type not in ['transformer', 'gpt2', 'rwkv']:
            args.n_embd = best_params.get('embed_size', args.n_embd)
        
        args.n_layer = best_params.get('layer_count', args.n_layer)
        
        # Set momentum if applicable
        if args.optim in ["sgdmomentum", "rmsprop"] and 'momentum' in best_params:
            args.momentum = best_params['momentum']
        else:
            args.momentum = 0.0  # Default momentum
        
        # Set beta1 and beta2 if applicable
        if args.optim in [
            "adam", "lamb", "adamhd", "lambhd", "gflambhd",
            "gflamb", "adamp", "adabelief", "adamz", "caadam",
            "warpadam", "adopt", "bgeadam", "snradam", 'adamax'
        ] and ('beta1' in best_params and 'beta2' in best_params):
            args.beta1 = best_params['beta1']
            args.beta2 = best_params['beta2']
        else:
            args.beta1 = 0.9  # Default beta1
            args.beta2 = 0.999  # Default beta2
        
        # Set hypergrad_lr if applicable
        if args.optim in ["adamhd", "lambhd"]:
            args.hypergrad_lr = best_params.get('hyper_lr', args.hypergrad_lr)
        else:
            args.hypergrad_lr = 1e-8  # Default hypergrad_lr
        
        # Set alpha for RMSprop or other optimizers
        if args.optim == "rmsprop":
            args.alpha = best_params.get('alpha', 0.9)
        else:
            args.alpha = 0.9  # Default alpha
        
        # Set Adagrad specific parameters
        if args.optim == "adagrad":
            args.initial_accumulator_value = best_params.get('init_accum_val', 1e-4)
            args.lr_decay = best_params.get('lrdecay', 0.0)
        else:
            args.initial_accumulator_value = 0.0  # Default value
            args.lr_decay = 0.0  # Default value
        
        # Set GrokFast specific parameters
        if args.optim in ["gflamb", "gfadamw"]:
            args.grokfast_alpha = best_params.get('gf_alpha', 0.5)
        else:
            args.grokfast_alpha = 0.0  # Default value
        
        # Set cautious_factor for gflambhd
        if args.optim == "gflambhd":
            args.cautious_factor = best_params.get('caut_fact', 1e-18)
        else:
            args.cautious_factor = 0.0  # Default value
        
        # Set delta for adamp
        if args.optim == "adamp":
            args.delta = best_params.get('delta', 0.0)
        else:
            args.delta = 0.0  # Default value
        
        # Set eps for adabelief
        if args.optim == "adabelief":
            args.eps = best_params.get('eps', 1e-16)
        else:
            args.eps = 1e-16  # Default value
        
        # Set number of heads if transformer-like
        if args.type in ['transformer', 'gpt2', 'rwkv']:
            args.n_head = best_params.get('n_heads', args.n_head)
        else:
            args.n_head = 1  # Default to 1 if not transformer-like
        
        # Set hyperparameters for newly added optimizers
        if args.optim == "adamz":
            args.overshoot_factor = best_params.get('overshoot_factor', 0.5)
            args.stagnation_factor = best_params.get('stagnation_factor', 1.2)
            args.stagnation_threshold = best_params.get('stagnation_threshold', 0.2)
            args.patience = best_params.get('patience', 100)
            args.stagnation_period = best_params.get('stagnation_period', 10)
            args.max_grad_norm = best_params.get('max_grad_norm', 1.0)
            args.lr_bounds = (
                best_params.get('lr_min_bound', 1e-7),
                best_params.get('lr_max_bound', 1.0)
            )
        else:
            # Assign default values if not using AdamZ
            args.overshoot_factor = 0.5
            args.stagnation_factor = 1.2
            args.stagnation_threshold = 0.2
            args.patience = 100
            args.stagnation_period = 10
            args.max_grad_norm = 1.0
            args.lr_bounds = (1e-7, 1.0)
        
        if args.optim == "caadam":
            args.scaling_method = best_params.get('scaling_method', 'multiplicative')
            args.gamma = best_params.get('gamma', 0.95)
        else:
            args.scaling_method = 'multiplicative'
            args.gamma = 0.95
        
        if args.optim == "adopt":
            args.clip_value = best_params.get('clip_value', 0.1)
        else:
            args.clip_value = 0.1  # Default value
        
        if args.optim == "nsgda":
            args.layer_wise = best_params.get('layer_wise', True)
        else:
            args.layer_wise = True  # Default value
        
        if args.optim == "bgeadam":
            args.entropy_weight = best_params.get('entropy_weight', 0.01)
            args.beta1_max = best_params.get('beta1_max', 0.9)
            args.beta1_min = best_params.get('beta1_min', 0.5)
            args.beta2_max = best_params.get('beta2_max', 0.999)
            args.beta2_min = best_params.get('beta2_min', 0.9)
        else:
            args.entropy_weight = 0.01
            args.beta1_max = 0.9
            args.beta1_min = 0.5
            args.beta2_max = 0.999
            args.beta2_min = 0.9
        if args.type in ['transformer', 'gpt2', 'rwkv']:
            args.n_head = best_params.get('n_heads', args.n_head)
        else:
            args.n_head = 1

        # Save the best_params.txt again in case of optimization
        if args.optimize:
            with open(os.path.join(args.work_dir, 'best_params.txt'), 'w') as f:
                f.write(str(best_params))

        # Optionally save individual configurations
        if args.optimize or args.resume:
            if args.type not in ['transformer', 'gpt2', 'rwkv']:
                with open('conf_embed', 'w') as f:
                    f.write(str(best_params.get('embed_size', args.n_embd)))
            with open('conf_layer', 'w') as f:
                f.write(str(best_params.get('layer_count', args.n_layer)))
            with open('conf_type', 'w') as f:
                f.write(str(best_params.get('model', args.type)))
            with open('conf_head', 'w') as f:
                f.write(str(args.n_head))
            args.type = best_params.get('model', args.type)

    # Initialize the model with the best hyperparameters
    config = ModelConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        n_embd2=args.n_embd2
    )
    model = initialize_model(args, config)
    model.to(args.device)

    print(f"Model initialized with #params: {sum(p.numel() for p in model.parameters())}")

    if args.resume or args.sample_only:
        print("Resuming from existing model in the workdir")
        model_path = os.path.join(args.work_dir, 'model.pt')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded model state from {model_path}")
        else:
            print(f"Error: {model_path} does not exist. Cannot resume without the model.")
            sys.exit(1)

    if args.sample_only:
        do_sample = bool(args.do_sample == 1)
        prompt = None if args.prompt == "AHOGALOPAKURA" else args.prompt
        print_samples(args, num=args.num_samples, prompt=prompt, do_sample=do_sample, steper=args.steps_sample)
        sys.exit()

    # Initialize the optimizer with loaded or default parameters
    optimizer = initialize_optimizer(
            model, args,
            alpha=args.alpha,
            beta1=args.beta1,
            beta2=args.beta2,
            momentum=args.momentum,
            hypergrad_lr=args.hypergrad_lr,
            grokfast_alpha=args.grokfast_alpha,
            cautious_factor=args.cautious_factor,
            initial_accumulator_value=args.initial_accumulator_value,
            lr_decay=args.lr_decay,
            delta=args.delta,
            eps=args.eps,
            overshoot_factor=args.overshoot_factor,
            stagnation_factor=args.stagnation_factor,
            stagnation_threshold=args.stagnation_threshold,
            patience=args.patience,
            stagnation_period=args.stagnation_period,
            max_grad_norm=args.max_grad_norm,
            lr_bounds=args.lr_bounds,
            scaling_method=args.scaling_method,
            gamma=args.gamma,
            clip_value=args.clip_value,
            layer_wise=args.layer_wise,
            entropy_weight=args.entropy_weight,
            beta1_max=args.beta1_max,
            beta1_min=args.beta1_min,
            beta2_max=args.beta2_max,
            beta2_min=args.beta2_min
        )

    # Optionally load the optimizer state if resuming
    if args.resume:
        optimizer_path = os.path.join(args.work_dir, 'optimizer.pt')
        if os.path.exists(optimizer_path):
            load_optimizer_state(optimizer, optimizer_path)
            print(f"Loaded optimizer state from {optimizer_path}")
        else:
            print(f"Warning: {optimizer_path} does not exist. Starting optimizer from scratch.")

    # Training loop
    best_loss = None
    step = 0

    batch_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)
    trainsamp = 10 if args.line_delim else 3
    trainsteper = args.seq_len * 2 if not args.line_delim else 100#100 if args.seq_len < 100 else args.seq_len
    while True:
        t0 = time.time()

        # Get the next batch, ship to device, and unpack it to input and target
        batch = batch_loader.next()
        batch = [t.to(args.device) for t in batch]
        X, Y = batch

        # Feed into the model
        logits, loss = model(X, Y)

        # Calculate the gradient, update the weights
        model.zero_grad(set_to_none=True)
        loss.backward()
        # grads = gradfilter_ema(model, grads=grads)  # Uncomment if using gradfilter_ema
        optimizer.step()

        # Wait for all CPU work on the GPU to finish then calculate iteration time taken
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        t1 = time.time()
        lr = optimizer.param_groups[0]['lr']
        #decay = optimizer.param_groups[0]['weight_decay']

        # Logging
        if step % 10 == 0:
            print(f"step {step} | loss {loss.item():.4f} | LR {lr:.9f} | step time {(t1 - t0) * 1000:.2f}ms")

        # Evaluate the model
        if step > 0 and step % 100 == 0:
            train_loss = evaluate(model, train_dataset, batch_size=10, max_batches=10)
            test_loss = evaluate(model, test_dataset, batch_size=10, max_batches=10)
            # writer.add_scalar("Loss/train", train_loss, step)
            # writer.add_scalar("Loss/test", test_loss, step)
            # writer.flush()
            print(f"step {step} train loss: {train_loss} test loss: {test_loss}")
            # save the model to disk if it has improved
            # scheduler.step(test_loss)
            if train_only == 1:
                if best_loss is None or train_loss < best_loss:
                    out_path = os.path.join(args.work_dir, "model.pt")
                    print(f"train loss {train_loss} is the best so far, saving model to {out_path}")
                    torch.save(model.state_dict(), out_path)
                    best_loss = train_loss
                    save_optimizer_state(optimizer, os.path.join(args.work_dir, "optimizer.pt"))
            else:
                if best_loss is None or test_loss < best_loss:
                    out_path = os.path.join(args.work_dir, "model.pt")
                    print(f"test loss {test_loss} is the best so far, saving model to {out_path}")
                    torch.save(model.state_dict(), out_path)
                    best_loss = test_loss
                    save_optimizer_state(optimizer, os.path.join(args.work_dir, "optimizer.pt"))

        # Sample from the model
        if step > 0 and step % 1000 == 0:
            print("Sampling with temp 1.0")
            print_samples(args, num=trainsamp, prompt=None, steper=trainsteper, temperature=1.0)
            print("Sampling with temp 0.8")
            print_samples(args, num=trainsamp, prompt=None, steper=trainsteper, temperature=0.8)
            print("Sampling with temp 0.5")
            print_samples(args, num=trainsamp, prompt=None, steper=trainsteper, temperature=0.5)
            print("Sampling with temp 0.2")
            print_samples(args, num=trainsamp, prompt=None, steper=trainsteper, temperature=0.2)

        step += 1
        # Termination conditions
        if args.max_steps >= 0 and step >= args.max_steps:
            break
