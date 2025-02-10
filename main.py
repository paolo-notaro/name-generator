import glob
import os
import unicodedata
from datetime import datetime

import matplotlib.pyplot as plt
import mlflow
import torch

from model import RNN, GRUModel, LSTMModel, MixtureOfExperts, TransformerModel
from parsing import VALID_ARCHITECTURES, parse_args
from train import all_letters, input_tensor, n_letters, to_one_hot, train


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


# Sample from a category and starting letter
def sample(rnn, category, all_categories, start_letter="A", max_length=20):
    with torch.no_grad():  # no need to track history in sampling
        cat_tensor = to_one_hot(category, all_categories)
        input_ = input_tensor(start_letter)
        hidden = rnn.init_hidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(cat_tensor, input_[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input_ = input_tensor(letter)

        return output_name


# Get multiple samples from one category and multiple starting letters
def samples(rnn, category, all_categories, start_letters="ABC"):
    for start_letter in start_letters:
        print(sample(rnn, category, all_categories, start_letter))


def main():
    args = parse_args()
    mlflow.set_tracking_uri("file:./mlruns")  # Local MLflow tracking

    print("Loading data...")
    category_lines, all_categories = load_files("data/names/*.txt")

    print("Preparing model...")

    # load model
    if args.model:
        model = torch.load(args.model, weights_only=False)
        model.eval()
    else:
        model = VALID_ARCHITECTURES[args.architecture](
            input_size=n_letters,
            hidden_size=args.hidden_size,
            output_size=n_letters,
            n_categories=len(all_categories),
        )

    if args.do_training:
        time_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        save_model_path = (
            args.save_model_path
            if args.save_model_path
            else f"models/{args.architecture}_{time_start}.pt"
        )
        print("Training model with MLflow tracking...")
        all_losses = train(model, args, all_categories, category_lines)

        torch.save(model, save_model_path)
        mlflow.log_artifact(save_model_path)

        plt.figure()
        plt.plot(all_losses)
        plt.title(f"Training Loss - {args.architecture}")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.savefig(f"images/loss_{args.architecture}_{time_start}.png")
        mlflow.log_artifact(f"images/loss_{args.architecture}_{time_start}.png")
        plt.show()

    print("Sampling names...")
    samples(
        model, all_categories=all_categories, category="Russian", start_letters="RUS"
    )
    samples(
        model, all_categories=all_categories, category="German", start_letters="GER"
    )
    samples(
        model, all_categories=all_categories, category="Spanish", start_letters="SPA"
    )
    samples(
        model, all_categories=all_categories, category="Chinese", start_letters="CHI"
    )
    samples(
        model, all_categories=all_categories, category="Italian", start_letters="DPS"
    )


if __name__ == "__main__":
    main()
