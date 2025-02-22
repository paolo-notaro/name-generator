"""training.py: Train the RNN model with MLflow logging."""

import time
from argparse import Namespace

import mlflow
import torch
import torch.nn.utils.rnn as rnn_utils
from torch.optim import Adam
from tqdm import tqdm

from utils import (
    VALID_LOSSES,
    input_tensor,
    random_choice,
    target_tensor,
    time_since,
    to_one_hot,
)


# Get a random category and random line from that category
def random_training_pair(all_categories, category_lines):
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    return category, line


# Make category, input, and target tensors from a random category, line pair
def random_training_example(all_categories, category_lines):
    category, line = random_training_pair(all_categories, category_lines)
    cat_tensor = to_one_hot(category, all_categories)
    input_line_tensor = input_tensor(line)
    target_line_tensor = target_tensor(line)
    return cat_tensor, input_line_tensor, target_line_tensor


def random_training_batch(all_categories, category_lines, batch_size):
    category_tensors = []
    input_line_tensors = []
    target_line_tensors = []

    for _ in range(batch_size):
        category, line = random_training_pair(all_categories, category_lines)
        category_tensors.append(to_one_hot(category, all_categories))
        input_line_tensors.append(input_tensor(line))
        target_line_tensors.append(target_tensor(line))

    category_tensor = torch.cat(category_tensors, dim=0)
    input_line_tensor = rnn_utils.pad_sequence(input_line_tensors, batch_first=True)
    target_line_tensor = rnn_utils.pad_sequence(target_line_tensors, batch_first=True)

    # repeat category tensors L times to match pad sequence length
    sequence_length = input_line_tensor.size(1)
    category_tensor = category_tensor.expand(
        batch_size, sequence_length, category_tensor.size(-1)
    )

    return category_tensor, input_line_tensor, target_line_tensor


def train_iteration(
    model,
    optimizer,
    criterion,
    category_tensors,
    input_line_tensors,
    target_line_tensors,
):
    batch_size = len(input_line_tensors)
    hidden = model.init_hidden(batch_size)

    model.zero_grad()

    output, hidden = model(category_tensors, input_line_tensors, hidden)
    loss = criterion(output.view(-1, output.size(-1)), target_line_tensors.view(-1))

    loss.backward()
    optimizer.step()

    return output, loss.item()


def train(model, args: Namespace, all_categories, category_lines):
    """Train the model while logging with MLflow."""

    mlflow.set_experiment(args.experiment_name)
    model.train()

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("architecture", model.__class__.__name__)
        mlflow.log_param("hidden_size", args.hidden_size)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("n_iterations", args.n_iterations)
        mlflow.log_param("loss_type", args.criterion)
        mlflow.log_param("batch_size", args.batch_size)

        all_losses = []
        total_loss = 0.0
        loss_fn = VALID_LOSSES[args.criterion]
        optimizer = Adam(model.parameters(), lr=args.learning_rate)

        start = time.time()
        for i in (pbar := tqdm(range(1, args.n_iterations + 1))):
            category_tensors, input_line_tensors, target_line_tensors = (
                random_training_batch(all_categories, category_lines, args.batch_size)
            )
            _, loss = train_iteration(
                model,
                optimizer,
                loss_fn,
                category_tensors,
                input_line_tensors,
                target_line_tensors,
            )
            total_loss += loss
            mlflow.log_metric("loss", loss, step=i)

            pbar.set_description(f"{loss=:6.4f}")

            if i % args.plot_every == 0:
                avg_loss = total_loss / args.plot_every
                all_losses.append(avg_loss)
                mlflow.log_metric("avg_loss", avg_loss, step=i)
                total_loss = 0

        training_time = time.time() - start
        mlflow.log_metric("training_time", training_time)
        model.eval()

        print(f"Training complete: {training_time:.2f}s. Saving model...")

        return all_losses
