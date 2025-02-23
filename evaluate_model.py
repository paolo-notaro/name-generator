from argparse import ArgumentParser

import torch

from model import RNN, GRUModel, LSTMModel, MixtureOfExperts, TransformerModel
from utils import (
    all_letters,
    input_tensor,
    load_files,
    n_letters,
    to_one_hot,
)


# Sample from a category and starting letter
def sample(model, category, all_categories, start_letter="A", max_length=20):
    with torch.no_grad():  # no need to track history in sampling
        cat_tensor = to_one_hot(category, all_categories)
        input_ = input_tensor(start_letter)
        hidden = model.init_hidden(batch_size=1)

        output_name = start_letter

        for _ in range(max_length):
            output, hidden = model(cat_tensor, input_.unsqueeze(0), hidden)
            _, top_i = output.topk(1, dim=-1)
            top_i = top_i[:, -1, :].squeeze().item()

            # end of generation check
            if top_i == n_letters - 1:
                break

            letter = all_letters[top_i]
            output_name += letter

            if isinstance(model, (RNN, LSTMModel, GRUModel)):
                # we only give the next letter as input, hidden memorizes the previous ones
                input_ = input_tensor(letter)
            elif isinstance(model, (TransformerModel, MixtureOfExperts)):
                # we give the whole name as input
                input_ = input_tensor(output_name)
                cat_tensor = torch.cat(
                    [cat_tensor, to_one_hot(category, all_categories)], dim=1
                )

        return output_name


# Get multiple samples from one category and multiple starting letters
def samples(model, category, all_categories, start_letters="ABC"):
    for start_letter in start_letters:
        print(sample(model, category, all_categories, start_letter))


def parse_evaluate_args():
    parser = ArgumentParser(description="Evaluate the generative model.")
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Path to a pre-trained model"
    )
    return parser.parse_args()


def main():
    args = parse_evaluate_args()

    print("Loading data...")
    _, all_categories = load_files("data/names/*.txt")

    # load model
    print("Loading model...")
    model = torch.load(args.model, weights_only=False)
    model.eval()  # Ensure the model is in evaluation mode

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
