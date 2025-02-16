import glob
import os
import random
import string
import time
import unicodedata
from math import floor

import torch
from torch.nn import CrossEntropyLoss, L1Loss, MSELoss, NLLLoss

from model import RNN, GRUModel, LSTMModel, MixtureOfExperts, TransformerModel

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1  # plus EOS marker


def time_since(since):
    now = time.time()
    s = now - since
    m = floor(s / 60)
    s -= m * 60
    return "%3dm %2ds" % (m, s)


def find_files(path):
    return glob.glob(path)


def unicode_to_ascii(s):
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in all_letters
    )


# Read a file and split into lines
def read_lines(file_path: str):
    lines = open(file_path, encoding="utf-8").read().strip().split("\n")
    return [unicode_to_ascii(line) for line in lines]


def load_files(names_data_path: str):
    # Build the category_lines dictionary, a list of lines per category
    category_lines = {}
    all_categories = []
    for filename in find_files(names_data_path):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = read_lines(filename)
        category_lines[category] = lines

    if len(all_categories) == 0:
        raise RuntimeError(
            "Data not found. Make sure that you downloaded data "
            "from https://download.pytorch.org/tutorial/data.zip and extract it to "
            "the current directory."
        )

    return category_lines, all_categories


VALID_LOSSES = {
    "nll": NLLLoss(),
    "ce": CrossEntropyLoss(),
    "mse": MSELoss(),
    "l1": L1Loss(),
}

VALID_ARCHITECTURES = {
    "rnn": RNN,
    "lstm": LSTMModel,
    "gru": GRUModel,
    "transformer": TransformerModel,
    "moe": MixtureOfExperts,
}


# Generate random experiment name for MLFlow
def random_experiment_name():
    """Generate a random experiment name."""
    return "exp-" + "".join(random.choices(string.ascii_lowercase + string.digits, k=8))


# Random item from a list
def random_choice(lst: list):
    return lst[random.randint(0, len(lst) - 1)]


# One-hot vector for category
def to_one_hot(category, all_categories):
    li = all_categories.index(category)
    one_hot = torch.zeros(1, len(all_categories))
    one_hot[0][li] = 1
    return one_hot


# One-hot matrix of first to last letters (not including EOS) for input
def input_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor


# LongTensor of second letter to end (EOS) for target
def target_tensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes)
