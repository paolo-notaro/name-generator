import random
import string
from argparse import ArgumentParser

from torch.nn import CrossEntropyLoss, L1Loss, MSELoss, NLLLoss

from model import RNN, GRUModel, LSTMModel, MixtureOfExperts, TransformerModel

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


def random_experiment_name():
    """Generate a random experiment name."""
    return "exp-" + "".join(random.choices(string.ascii_lowercase + string.digits, k=8))


def parse_args():

    parser = ArgumentParser(description="name_generator")

    # Model parameters
    parser.add_argument(
        "-m",
        "--model",
        help="Model file path",
        required=False,
        default="models/{architecture}",
    )
    parser.add_argument(
        "-a",
        "--architecture",
        help="Model architecture",
        required=False,
        default="rnn",
        choices=list(VALID_ARCHITECTURES.keys()),
        type=str,
    )
    parser.add_argument(
        "-l",
        "--load-model",
        help="If true, load model from file",
        required=False,
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-s",
        "--save-model-path",
        help="Model save path",
        required=False,
        default=None,
        type=str,
    )
    parser.add_argument(
        "--hidden-size",
        help="Hidden size of the RNN",
        required=False,
        default=128,
        type=int,
    )

    # Training parameters
    parser.add_argument(
        "-t",
        "--do-training",
        help="If true, do training",
        required=False,
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-n",
        "--n-iterations",
        help="Number of iterations",
        required=False,
        type=int,
        default=100_000,
    )
    parser.add_argument(
        "-p",
        "--print-every",
        help="Print every n iterations",
        required=False,
        default=500,
    )
    parser.add_argument(
        "-pe",
        "--plot-every",
        help="Plot every n iterations",
        required=False,
        default=500,
    )
    parser.add_argument(
        "--criterion",
        help="Criterion for the RNN",
        required=False,
        default="nll",
        choices=VALID_LOSSES.keys(),
        type=str,
    )
    parser.add_argument(
        "--learning-rate",
        help="Learning rate for the RNN",
        required=False,
        default=0.0005,
        type=float,
    )

    # MLflow experiment tracking
    parser.add_argument(
        "-e",
        "--experiment_name",
        type=str,
        default=random_experiment_name(),
        help="Experiment name for MLflow tracking. Defaults to a random name.",
    )

    return parser.parse_args()
