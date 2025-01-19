#!/bin/bash

# Activate the conda environment
eval "$(conda shell.bash hook)" # Allows the BASH script to change virtual conda environments.
conda activate pytorch # Activates the virtual conda environment.

# Function to display model types
display_model_types() {
    echo "Select Model Type:"
    echo "--------------------Models--------------------"
    echo "1) Bi-Gram (basic logits based lookup table)"
    echo "--------------------Multilayered Perceptrons--------------------"
    echo "2) Multilayered AdaLin (MLP with no act functions)"
    echo "3) Basic MultiLayered Perceptron with Sigmoidal activations, inspired by EasyNN (The most basic, original MLP I started with)"
    echo "4) Residual MultiLayered Perceptron with SiLU activation"
    echo "5) Just like 4), but with complex weights (usually as good as the previous MLP with doubled hidden size)"
    echo "6) Basic MLP with per-neuron trainable Piecewise Linears applied to outputs"
    echo "7) Transformer-like Residual MLP Block w/o attention and with trainable PWLs applied to each block's input and output"
    echo "--------------------Recurrent Neural Networks--------------------"
    echo "8) Basic Recurrent Neural Network with Residual Connections, LayerNorm and SiLU"
    echo "9) Independently Recurrent Neural Network with the Parametric ReLU Activation"
    echo "10) ReZero Recurrent Unit, from (Gates are not what you need)"
    echo "11) IndyGRU ported over from TensorFlow 1.15.x"
    echo "12) Light Recurrent Unit (The best RNN I've got on many tasks)"
    echo "13) Gated Recurrent Unit (using nn.GRUCells)"
    echo "14) Long-Short Term Memory Cell (LSTM, using nn.LSTMCells)"
    echo "--------------------Transformers and its alternatives--------------------"
    echo "15) 4-layered MLP with a Causal Bag of Words"
    echo "16) GPT-2 Transformer (Normal GPT-2, has trainable positional embeddings)"
    echo "17) GPT-2 Transformer with Parabolic Cone Activations (has trainable positional embeddings)"
    echo "18) Receptance-Weight Key Value model, fully parallel RNN (The best Transformer-like model I've got yet)"
    echo "19) MinGRU (Fully Parallel GRU, from 'Were RNNs all we needed?') (GRU with full parallelization, kinda unstable training, weird loss issue when seq_len's of examples differ)"
    echo "20) A Temporal Convnet (Surprisingly good generalization on smaller datasets, seems to be the best of MLP (strong positional biases) and Transformers (being able to properly deal with sequential data to an extent)"
    echo "21) Hyper Mixer model (A fully MLP based language model)"
    echo "--------------------AutoSearch--------------------"
    echo "22) Grid Search best model (Deprecated)"
    echo "--------------------OPTUNA--------------------"
    echo "23) All versions of the MLP I've got"
    echo "24) All versions of the RNN I've got"
    echo "25) All transformer variants, including RWKV"
    echo "26) All models (No MinGRU, for line-delimited, due to the weird loss issue being a nonfactor in regular files with a fixed seq_len)"
    echo "27) All models (including MinGRU, either for files with similarly long examples, or regular files)"
    echo "--------------------Time to choose...--------------------"
}

# Function to display optimizer types
display_optimizer_types() {
    echo "Select Optimizer Type:"
    echo "1) Basic Stochastic Gradient Descent (Converges decently well, has a habit of diverging, not great with sparse gradients)"
    echo "2) Stochastic Gradient Descent with Momentum (The GOAT for generalization, also has a habit of diverging if left alone long enough)"
    echo "3) Root Mean Square (RMS)prop (applies a moving average for smoother updates)"
    echo "4) Adagrad (adapts learning rate, known for a vanishing LR issue)"
    echo "5) Adam (The GOAT for many tasks, doesn't need much hyperparam optim)"
    echo "6) Adam with Hypergradient Descent (Kinda weird, sometimes performs better than Adam, sometimes it's even worse than base SGD)"
    echo "7) Lamb (Each layer gets its own learning rate + a trust ratio, a VERY strong optimizer, really good results no matter the architecture (besides maybe Bigram), the best optimizer for RNNs strangely enough)"
    echo "8) Lamb with Hypergradient Descent (Extremely unstable, not very good)"
    echo "9) GrokFast AdamW (Strong option for small batch sizes due to EMA-based smoothing of gradients, supposed to accelerate Grokking)"
    echo "10) GrokFast Lamb (Needs more testing)"
    echo "11) Adam Atan2 (Adam using a math function instead of an eps parameter, has some extra regularizations)"
    echo "12) AdamP (Fairly slow, but awesome for GANs, seems good on this as well)"
    echo "13) AdaBelief (Needs tuning for Optuna, likes to diverge like SGD, probably cuz Optuna keeps choosing a high eps, otherwise a strong choice. eps value sets whether the optimizer is closer to SGD (when high) or Adam (when low))"
    # Adding new optimizers
    echo "14) AdamZ (Adam optimizer variant with overshoot and stagnation control)"
    echo "15) CaAdam (Context-aware Adam optimizer with scaling methods)"
    echo "16) WarpAdam (Adam optimizer with weight projections)"
    echo "17) ADOPT (Adaptive Optimizer with gradient clipping, unstable)"
    echo "18) NSGDA (Normalized Stochastic Gradient Descent Ascent optimizer, implementation seems off as it doesn't work well, not even in GANs)"
    echo "19) BGEAdam (Adam optimizer with Balanced Gradient Estimation)"
    echo "20) SNRAdam (Signal-to-Noise Ratio based Adam optimizer)"
    echo "21) Adamax (Adam with max-norm over gradients)"
    echo "22) AdaDelta (Adagrad variant that alleges to get rid of vanishing learning rates, and allegedly doesn't require a learning rate)"
}

# Prompt for mode selection
echo "Select Mode:"
echo "1) Train"
echo "2) Continue Training"
echo "3) Sample"
read -p "Enter your choice (1/2/3): " modo

if [[ $modo -eq 1 ]]
then
    # Mode 1: Training
    read -p "Input file: " intake

    # Check if the input file exists
    if [[ ! -f "$intake" ]]; then
        echo "Error: Input file '$intake' does not exist."
        exit 1
    fi

    # Print the absolute path of the input file for debugging
    if command -v realpath >/dev/null 2>&1; then
        input_path=$(realpath "$intake")
    elif command -v readlink >/dev/null 2>&1; then
        input_path=$(readlink -f "$intake")
    else
        input_path="$intake" # Fallback if realpath/readlink are not available
    fi
    echo "Input file path: $input_path"

    # Prompt to check if the input file is line-delimited
    read -p "Is the input file line-delimited? (y/n): " is_line_delim

    if [[ "$is_line_delim" =~ ^[Yy]$ ]]
    then
        line_delim=1
        seq_len=0
    elif [[ "$is_line_delim" =~ ^[Nn]$ ]]
    then
        line_delim=0
        read -p "Enter sequence length (seq_len): " seq_len
        # Validate that seq_len is a positive integer
        if ! [[ "$seq_len" =~ ^[0-9]+$ ]]; then
            echo "Error: Sequence length must be a positive integer."
            exit 1
        fi
    else
        echo "Invalid input for line-delimited option. Please enter 'y' or 'n'."
        exit 1
    fi

    # Prompt for model type
    display_model_types
    read -p "Enter your choice (1-27): " gulag

    # Prompt for embedding size
    read -p "Embed size: " embed1

    # Prompt for layer count
    read -p "Layer count: " layercount

    # Determine model type and number of heads based on user input
    case $gulag in
        1)
            typo="bigram"
            nhead=1
            ;;
        2)
            typo="mlp_adalin"
            nhead=1
            ;;
        3)
            typo="mlp_og"
            nhead=1
            ;;
        4)
            typo="mlp_lm"
            nhead=1
            ;;
        5)
            typo="mlp_pca"
            nhead=1
            ;;
        6)
            typo="mlp_gelu"
            nhead=1
            ;;
        7)
            typo="mlp"
            nhead=1
            ;;
        8)
            typo="ogrnn"
            nhead=1
            ;;
        9)
            typo="rnn"
            nhead=1
            ;;
        10)
            typo="rru"
            nhead=1
            ;;
        11)
            typo="gru"
            nhead=1
            ;;
        12)
            typo="moglstm"
            nhead=1
            ;;
        13)
            typo="oggru"
            nhead=1
            ;;
        14)
            typo="lstm"
            nhead=1
            ;;
        15)
            typo="bow"
            nhead=1
            ;;
        16)
            typo="gpt2"
            read -p "Head count: " nhead
            ;;
        17)
            typo="transformer"
            read -p "Head count: " nhead
            ;;
        18)
            typo="rwkv"
            read -p "Head count: " nhead
            ;;
        19)
            typo="mingru"
            nhead=1
            ;;
        20)
            typo="convnet"
            nhead=1
            ;;
        21)
            typo="hypermixer"
            read -p "Head count: " nhead
            ;;
        22)
            typo="auto"
            echo "Which model category to benchmark?"
            echo "1) All"
            echo "2) No RNNs"
            echo "3) No MLPs"
            echo "4) No Transformers"
            echo "5) RNN-only"
            echo "6) Transformer-only"
            echo "7) MLP-only"
            read -p "Enter your choice (1-7): " benchy
            case $benchy in
                1) typer='all' ;;
                2) typer='no-rnns' ;;
                3) typer='no-mlps' ;;
                4) typer='no-transformers' ;;
                5) typer='rnn-only' ;;
                6) typer='transformer-only' ;;
                7) typer='mlp-only' ;;
                *) echo "Wrong input for model category."; exit 1 ;;
            esac
            read -p "Step size: " stecount
            ;;
        23)
            typo="automlp"
            nhead=1
            ;;
        24)
            typo="autornn"
            nhead=1
            ;;
        25)
            typo="transformerlikes"
            read -p "Head count: " nhead
            ;;
        26)
            typo="autoallnomingru"
            read -p "Head count: " nhead
            ;;
        27)
            typo="autoall"
            read -p "Head count: " nhead
            ;;
        *)
            echo "Wrong model type selected."
            exit 1
            ;;
    esac
    typer='all'
    stecount=200
    fi

    # Prompt for optimizer type
    display_optimizer_types
    read -p "Enter your choice (1-22): " optim_type

    # Map optimizer selection to optimizer names using case
    case $optim_type in
        1) optim="sgd" ;;
        2) optim="sgdmomentum" ;;
        3) optim="rmsprop" ;;
        4) optim="adagrad" ;;
        5) optim="adam" ;;
        6) optim="adamhd" ;;
        7) optim="lamb" ;;
        8) optim="lambhd" ;;
        9) optim="gfadamw" ;;
        10) optim="gflamb" ;;
        11) optim="gflambhd" ;;
        12) optim="adamp" ;;
        13) optim="adabelief" ;;
        # New Optimizers
        14) optim="adamz" ;;
        15) optim="caadam" ;;
        16) optim="warpadam" ;;
        17) optim="adopt" ;;
        18) optim="nsgda" ;;
        19) optim="bgeadam" ;;
        20) optim="snradam" ;;
        21) optim="adamax" ;;
        22) optim="adadelta" ;;
        *) echo "Wrong optimizer type selected."; exit 1 ;;
    esac

    # Prompt to decide what to optimize
    read -p "Shall we optimize the model? (1=yes|0=no): " optimi
    if [[ $optimi -eq 1 ]]
    then
        optuna='model'
    else
        optuna='optim'
    fi

    # Prompt for batch size
    read -p "Batch size: " bath
    read -p "Train only? (1=yes|0=no): " tarain

    # Save configurations to files
    echo "${typo}" > conf_type
    echo "${layercount}" > conf_layer
    echo "${nhead}" > conf_head
    echo "${embed1}" > conf_embed
    echo "${intake}" > conf_dataset
    echo "${optim}" > conf_optim
    echo "${line_delim}" > conf_line_delim
    echo "${seq_len}" > conf_seq_len

    # Construct the Python command with conditional flags
    python_cmd="python makemore.py --input-file \"${intake}\" --type \"${typo}\" --n-layer \"${layercount}\" --n-head \"${nhead}\" --n-embd \"${embed1}\" --n-embd2 \"${embed1}\" --batch-size \"${bath}\" --optim \"${optim}\" --autotype \"${typer}\" --stepscount \"${stecount}\" --optimize --optimi \"${optuna}\" --train_only \"${tarain}\""

    # Add --line_delim if line_delim is 1
    if [[ $line_delim -eq 1 ]]
    then
        python_cmd+=" --line_delim"
    fi

    # Add --seq_len if line_delim is 0
    if [[ $line_delim -eq 0 ]]
    then
        python_cmd+=" --seq_len \"${seq_len}\""
    fi

    # Execute the Python command
    eval $python_cmd

elif [[ $modo -eq 2 ]]
then
    # Mode 2: Continue Training

    # Check if configuration files exist
    required_configs=("conf_type" "conf_layer" "conf_head" "conf_embed" "conf_dataset" "conf_optim" "conf_line_delim" "conf_seq_len")
    for config in "${required_configs[@]}"; do
        if [[ ! -f "$config" ]]; then
            echo "Error: Configuration file '$config' is missing. Cannot continue training."
            exit 1
        fi
    done

    typo=$(< conf_type)
    layercount=$(< conf_layer)
    nhead=$(< conf_head)
    embed1=$(< conf_embed)
    intake=$(< conf_dataset)
    optim=$(< conf_optim)
    line_delim=$(< conf_line_delim)
    seq_len=$(< conf_seq_len)

    # Print the input file path for debugging
    if command -v realpath >/dev/null 2>&1; then
        input_path=$(realpath "$intake")
    elif command -v readlink >/dev/null 2>&1; then
        input_path=$(readlink -f "$intake")
    else
        input_path="$intake" # Fallback if realpath/readlink are not available
    fi
    echo "Input file path: $input_path"

    # If the input file is not line-delimited, prompt for seq_len again
    if [[ $line_delim -eq 0 ]]
    then
        read -p "Enter sequence length (seq_len): " seq_len
        # Validate that seq_len is a positive integer
        if ! [[ "$seq_len" =~ ^[0-9]+$ ]]; then
            echo "Error: Sequence length must be a positive integer."
            exit 1
        fi
        # Update the configuration file with the new seq_len
        echo "${seq_len}" > conf_seq_len
    fi

    read -p "Batch size: " bath
    read -p "Train only? (1=yes|0=no): " tarain

    # Construct the Python command with conditional flags
    python_cmd="python makemore.py --input-file \"${intake}\" --type \"${typo}\" --n-layer \"${layercount}\" --n-head \"${nhead}\" --n-embd \"${embed1}\" --n-embd2 \"${embed1}\" --batch-size \"${bath}\" --resume --optim \"${optim}\" --train_only \"${tarain}\""

    # Add --line_delim if line_delim is 1
    if [[ $line_delim -eq 1 ]]
    then
        python_cmd+=" --line_delim"
    fi

    # Add --seq_len if line_delim is 0
    if [[ $line_delim -eq 0 ]]
    then
        python_cmd+=" --seq_len \"${seq_len}\""
    fi

    # Execute the Python command
    eval $python_cmd

elif [[ $modo -eq 3 ]]
then
    # Mode 3: Sampling

    # Check if configuration files exist
    required_configs=("conf_type" "conf_layer" "conf_head" "conf_embed" "conf_dataset" "conf_line_delim" "conf_seq_len")
    for config in "${required_configs[@]}"; do
        if [[ ! -f "$config" ]]; then
            echo "Error: Configuration file '$config' is missing. Cannot perform sampling."
            exit 1
        fi
    done

    typo=$(< conf_type)
    layercount=$(< conf_layer)
    nhead=$(< conf_head)
    embed1=$(< conf_embed)
    intake=$(< conf_dataset)
    line_delim=$(< conf_line_delim)
    seq_len=$(< conf_seq_len)

    # Print the input file path for debugging
    if command -v realpath >/dev/null 2>&1; then
        input_path=$(realpath "$intake")
    elif command -v readlink >/dev/null 2>&1; then
        input_path=$(readlink -f "$intake")
    else
        input_path="$intake" # Fallback if realpath/readlink are not available
    fi
    echo "Input file path: $input_path"

    read -p "Prompt: " prmopt
    read -p "Sample len: " samplelen
    read -p "Sample count: " samplecount

    if [[ -z "$prmopt" ]]
    then
        prmopt="AHOGALOPAKURA"
    fi

    if [[ $samplecount -eq 0 ]]
    then
        samplecount=1
        dosample=0
    else
        dosample=1
    fi

    # Construct the Python command with conditional flags
    python_cmd="python makemore.py --input-file \"${intake}\" --type \"${typo}\" --n-layer \"${layercount}\" --n-head \"${nhead}\" --n-embd \"${embed1}\" --n-embd2 \"${embed1}\" --sample-only --num-samples \"${samplecount}\" --prompt \"${prmopt}\" --do-sample \"${dosample}\" --steps_sample \"${samplelen}\""

    # Add --line_delim if line_delim is 1
    if [[ $line_delim -eq 1 ]]
    then
        python_cmd+=" --line_delim"
    fi

    # Add --seq_len if line_delim is 0
    if [[ $line_delim -eq 0 ]]
    then
        python_cmd+=" --seq_len \"${seq_len}\""
    fi

    # Execute the Python command
    eval $python_cmd

else
    echo "Wrong mode selected."
    exit 1
fi
