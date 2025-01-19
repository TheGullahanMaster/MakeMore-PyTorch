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
- mymodels.py: All model definitions
- lamb.py: Contains all nonbuilt-in optimizers used in the script
- makemorevis.html: A JavaScript based visualizer of activations for each token

# Credits
lamb.py: Modified from https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py
Original MakeMore script: Andrej Karpathy
OpenAI, Google, Anthropic: Their AI tools were used for some of these models, like the Temporal Convnet, and the optuna implemntation
RWKV-v5: Unknown github user, will update once found
