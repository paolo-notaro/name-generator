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


def parse_args():

    parser = ArgumentParser(description="name_generator")

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
        choices=list(VALID_ARCHITECTURES.values()),
        type=VALID_ARCHITECTURES.get,
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
        "-n",
        "--n-iterations",
        help="Number of iterations",
        required=False,
        default=100000,
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
        "-t",
        "--do-training",
        help="If true, do training",
        required=False,
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--hidden-size",
        help="Hidden size of the RNN",
        required=False,
        default=128,
        type=int,
    )

    parser.add_argument(
        "--criterion",
        help="Criterion for the RNN",
        required=False,
        default="nll",
        choices=VALID_LOSSES.values(),
        type=VALID_LOSSES.get,
    )
    parser.add_argument(
        "--learning-rate",
        help="Learning rate for the RNN",
        required=False,
        default=0.0005,
        type=float,
    )
    return parser.parse_args()
