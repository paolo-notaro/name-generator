from argparse import ArgumentParser

import torch

from utils import (
    VALID_ARCHITECTURES,
    all_letters,
    input_tensor,
    load_files,
    n_letters,
    to_one_hot,
)


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


def parse_evaluate_args():
    parser = ArgumentParser(description="Evaluate the generative model.")
    parser.add_argument(
        "--hidden_size", type=int, default=128, help="Size of RNN hidden layer"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to a pre-trained model"
    )
    return parser.parse_args()


def main():
    args = parse_evaluate_args()

    print("Loading data...")
    _, all_categories = load_files("data/names/*.txt")

    # load model
    print("Loading model...")
    model = torch.load(args.model, weights_only=False)
    model.eval()

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
