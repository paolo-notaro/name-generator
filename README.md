# Name Generator
## Description
A Python-based name generator that creates random names based on predefined patterns and rules.
Adapted from [this Pytorch tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html).

## Features
- Generate random names
- Customize name patterns
- Save generated names to a file

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/name-generator.git
    ```
2. Navigate to the project directory:
    ```bash
    cd name-generator
    ```
3. Install the required dependencies:
    ```bash
    poetry install
    ```

## Usage

### Train a Model

To train a model and generate a new name, use the following command:

```sh
python train_model.py -a <architecture> -e <experiment_name> --hidden_size <hidden_size> --learning_rate <learning_rate> --n_iterations <n_iterations> --print_every <print_every> --plot_every <plot_every> --criterion <criterion> --save_model_path <save_model_path>
```

Example:

```sh
python train_model.py -a lstm -e my_experiment --hidden_size 128 --learning_rate 0.005 --n_iterations 100000 --print_every 5000 --plot_every 1000 --criterion nll --save_model_path models/lstm_model.pt
```

### Evaluate a Model

To evaluate a pre-trained model, use the following command:

```sh
python evaluate_model.py --model <model_path> --hidden_size <hidden_size>
```

Example:

```sh
python evaluate_model.py --model models/lstm_model.pt --hidden_size 128
```

Full usage:

```
> python train_model.py -h

usage: train_model.py [-h] [-e EXPERIMENT_NAME] [--hidden_size HIDDEN_SIZE] [--learning_rate LEARNING_RATE] [--n_iterations N_ITERATIONS] [--print_every PRINT_EVERY] [--plot_every PLOT_EVERY] [--criterion {nll,ce,mse,l1}] [--save_model_path SAVE_MODEL_PATH]

Train the RNN model with MLflow logging.

options:
  -h, --help            show this help message and exit
  -e EXPERIMENT_NAME, --experiment_name EXPERIMENT_NAME
                        MLflow experiment name
  --hidden_size HIDDEN_SIZE
                        Size of RNN hidden layer
  --learning_rate LEARNING_RATE
                        Learning rate
  --n_iterations N_ITERATIONS
                        Number of training iterations
  --print_every PRINT_EVERY
                        Print progress every N iterations
  --plot_every PLOT_EVERY
                        Plot loss every N iterations
  --criterion {nll,ce,mse,l1}
                        Loss function
  --save_model_path SAVE_MODEL_PATH
                        Path to save the trained model

> python evaluate_model.py -h

usage: evaluate_model.py [-h] [--hidden_size HIDDEN_SIZE] --model MODEL

Evaluate the generative model.

options:
  -h, --help            show this help message and exit
  --hidden_size HIDDEN_SIZE
                        Size of RNN hidden layer
  --model MODEL         Path to a pre-trained model
```