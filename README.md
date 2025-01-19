# MakeMore-PyTorch
Karpathy's MakeMore script, modified to suit my personal needs. Can be used to train on line delimited and regular text files. It is my own collection of ideas either repurposed from elsewhere, or created using AI tools (Mostly ChatGPT)

# Dependencies
```
torch
optuna
pau (https://github.com/ChristophReich1996/Pade-Activation-Unit)
torchpwl
beartypes
einops
numpy
conda
```

# Usage
Run ```linegen.sh```, and follow instructions on terminal. You can choose from a variety of models and optimizers, and you can select the ```seq_len```, hidden dims, layer count and ```batch_size```.
Sampling and training resuming is also supported

# Files
- linegen.sh: The BASH script used to more accesibly run the script.
- makemore.py: The main training script, housing the training, Optuna optimization and sampling logic
- mymodels.py: All model definitions. Contains:
|- Bigram
|- "ADALIN", an MLP variant with no activation function
|- A simple sigmoidal MLP
|- A residual MLP with SiLU activations and layernorms
|- A complex valued variant of the above
|- A simple, non-residual MLP with trainable PieceWise Linear activation functions
|- A transformer-like residual MLP, w/o attention, and with trainable PieceWise Linears
Residual RNNs with zoneout and variational dropout:
|- Basic RNN cell with SiLU activation
|- IndRNN cell with PReLUs
|- RRU cell, from Gates are not what you need
|- IndyGRU with SiLU activations
|- LRU (Light Recurrent Unit) cell, only uses a forget gate
|- Builtin basic GRU cell
|- Builtin basic LSTM cell
"Transformer-likes", they (usually) have superior performance than the previous models
|- MLP with an "attention-like" causal Bag of Words
|- GPT-2 Transformer architecture
|- Same as above, but with Parabolic Cone activations
|- RWKV v5 model, does not yet have the recurrent sampling component
|- minGRU, a GRU variant designed to run with Parallel Scan
|- A Temporal Convnet designed by ChatGPT
|- Hyper Mixer model (wip, loss lower than any other model, but garbage sampling quality)
Deprecated and soon to be removed "Grid Search"
Optuna options (If chosen, Optuna will gain the ability to try multiple preset model options):
|- All MLP variants
|- All RNN variants
|- All models outside of minGRU
|- All models including minGRU
- lamb.py: Contains all nonbuilt-in optimizers used in the script
- makemorevis.html: A JavaScript based visualizer of activations for each token

# Sample datasets
- names.txt: A set of baby names included with the original script
- reverse_letter_order.txt: A simple dataset; 7 randomly generated letters, a comma, and the preceeding sequence mirrored. Most models can handle this fine, despite the script not being made for seq2seq
- ShakeSpeare.txt: A standard text file containing all Shakespeare works. A useful benchmark for smaller models.
- asciiart.txt: About 3 MBs of ASCII art crawled from the ascii art archive. A useful (albeit unorthodox) way to "benchmark" positional awareness and attention/memory. Needs a large seq_len due to the size of some ASCII arts

# Credits
lamb.py: Modified from https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py.

Original MakeMore script: Andrej Karpathy.

OpenAI, Google, Anthropic: Their AI tools were used for some of these models, like the Temporal Convnet, and the optuna implemntation.

RWKV-v5: Unknown github user, will update once found.
