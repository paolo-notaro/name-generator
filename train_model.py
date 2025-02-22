from argparse import ArgumentParser
from datetime import datetime

import matplotlib.pyplot as plt
import mlflow
import torch

from training import train
from utils import (
    VALID_ARCHITECTURES,
    VALID_LOSSES,
    load_files,
    n_letters,
    random_experiment_name,
)


def parse_train_args():
    parser = ArgumentParser(description="Train the RNN model with MLflow logging.")
    parser.add_argument(
        "-e",
        "--experiment_name",
        type=str,
        required=False,
        default=random_experiment_name(),
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=128, help="Size of RNN hidden layer"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.005, help="Learning rate"
    )
    parser.add_argument(
        "-n",
        "--n_iterations",
        type=int,
        default=100000,
        help="Number of training iterations",
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=1, help="Batch size for training"
    )
    parser.add_argument(
        "--plot_every", type=int, default=100, help="Plot loss every N iterations"
    )
    parser.add_argument(
        "--criterion",
        type=str,
        choices=VALID_LOSSES.keys(),
        default="nll",
        help="Loss function",
    )
    parser.add_argument(
        "-a",
        "--architecture",
        type=str,
        required=True,
        help="Model architecture",
        choices=VALID_ARCHITECTURES.keys(),
    )
    parser.add_argument("--model", type=str, help="Path to a pre-trained model")
    parser.add_argument(
        "--save_model_path", type=str, help="Path to save the trained model"
    )
    return parser.parse_args()


def plot_loss(all_losses: list[float], architecture: str, time_start: str):
    plt.figure()
    plt.plot(all_losses)
    plt.title(f"Training Loss - {architecture}")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig(f"images/loss_{architecture}_{time_start}.png")
    mlflow.log_artifact(f"images/loss_{architecture}_{time_start}.png")
    plt.show()


def main():
    args = parse_train_args()
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

    # start training
    time_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    save_model_path = (
        args.save_model_path
        if args.save_model_path
        else f"models/{args.architecture}_{time_start}.pt"
    )
    print("Training model with MLflow tracking...")
    all_losses = train(model, args, all_categories, category_lines)

    # save model
    torch.save(model, save_model_path)
    mlflow.log_artifact(save_model_path)

    # show loss plot
    plot_loss(all_losses, args.architecture, time_start)


if __name__ == "__main__":
    main()
