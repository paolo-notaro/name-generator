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

Train a model and generate a new name:

```
python main.py -t [...] #  train + inference
python main.py [...]    # inference only
```

Full usage:

```
> python man.py -h

usage: main.py [-h] [-m MODEL] [-a {rnn,lstm,gru,transformer,moe}] [-l] [-s SAVE_MODEL_PATH] [--hidden-size HIDDEN_SIZE] [-t] [-n N_ITERATIONS] [-p PRINT_EVERY] [-pe PLOT_EVERY]
               [--criterion {nll,ce,mse,l1}] [--learning-rate LEARNING_RATE] [-e EXPERIMENT_NAME]

name_generator

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Model file path
  -a {rnn,lstm,gru,transformer,moe}, --architecture {rnn,lstm,gru,transformer,moe}
                        Model architecture
  -l, --load-model      If true, load model from file
  -s SAVE_MODEL_PATH, --save-model-path SAVE_MODEL_PATH
                        Model save path
  --hidden-size HIDDEN_SIZE
                        Hidden size of the RNN
  -t, --do-training     If true, do training
  -n N_ITERATIONS, --n-iterations N_ITERATIONS
                        Number of iterations
  -p PRINT_EVERY, --print-every PRINT_EVERY
                        Print every n iterations
  -pe PLOT_EVERY, --plot-every PLOT_EVERY
                        Plot every n iterations
  --criterion {nll,ce,mse,l1}
                        Criterion for the RNN
  --learning-rate LEARNING_RATE
                        Learning rate for the RNN
  -e EXPERIMENT_NAME, --experiment_name EXPERIMENT_NAME
                        Experiment name for MLflow tracking. Defaults to a random name.
```