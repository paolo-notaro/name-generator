# Name Generator

## Description

Name Generator is a Python-based project that generates random names based on predefined patterns and rules. The project leverages various neural network architectures to create names that resemble those from different languages. This project is an extension [this Pytorch tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html).

## Models Involved

The project includes several neural network architectures to generate names:

- **RNN (Recurrent Neural Network)**: A simple recurrent network that processes sequences of characters.
- **LSTM (Long Short-Term Memory)**: An advanced RNN variant that can capture long-term dependencies.
- **GRU (Gated Recurrent Unit)**: Another RNN variant that is computationally efficient and effective for sequence modeling.
- **Transformer**: A model that uses self-attention mechanisms to process sequences in parallel, providing better performance for longer sequences.
- **Mixture of Experts**: A model that combines multiple expert networks to generate names, with a gating mechanism to select the most appropriate expert.

## Languages Considered

The Name Generator can generate names for various languages, including but not limited to:

- Russian
- German
- Spanish
- Chinese
- Italian

The project uses a dataset of names from these languages to train the models, ensuring that the generated names resemble real names from the respective languages.

## Features

- Generate random names based on different languages.
- Customize name patterns and generation rules.
- Save generated names to a file for further use.

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
python evaluate_model.py --model <model_path>
```

Example:

```sh
python evaluate_model.py --model models/lstm_model.pt
```

### Full usage

```
> python train_model.py -h

usage: train_model.py [-h] [-a ARCHITECTURE] [-e EXPERIMENT_NAME] [--hidden_size HIDDEN_SIZE] [--learning_rate LEARNING_RATE] [--n_iterations N_ITERATIONS] [--print_every PRINT_EVERY] [--plot_every PLOT_EVERY] [--criterion {nll,ce,mse,l1}] [--batch_size BATCH_SIZE]

Train the RNN model with MLflow logging.

options:
  -h, --help            show this help message and exit
  -a ARCHITECTURE, --architecture ARCHITECTURE
                        Model architecture (rnn, lstm, gru, transformer, mixture_of_experts)
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
  --batch_size BATCH_SIZE
                        Batch size
```

```
> python evaluate_model.py -h

usage: evaluate_model.py [-h] --model MODEL

Evaluate the generative model.

options:
  -h, --help            show this help message and exit
  --model MODEL         Path to a pre-trained model
```