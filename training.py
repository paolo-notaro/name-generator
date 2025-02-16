"""training.py: Train the RNN model with MLflow logging."""

import time
from argparse import Namespace

import mlflow
from torch.optim import Adam
from tqdm import tqdm

from utils import VALID_LOSSES, input_tensor, random_choice, target_tensor, to_one_hot


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


def train_iteration(
    model, optimizer, criterion, category_tensor, input_line_tensor, target_line_tensor
):
    target_line_tensor.unsqueeze_(-1)
    hidden = model.init_hidden()

    model.zero_grad()

    loss = 0.0
    output = None
    for i in range(input_line_tensor.size(0)):
        output, hidden = model(category_tensor, input_line_tensor[i], hidden)
        loss_i = criterion(output, target_line_tensor[i])
        loss += loss_i

    loss.backward()
    optimizer.step()

    return output, loss.item() / input_line_tensor.size(0)


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

        all_losses = []
        total_loss = 0
        loss_fn = VALID_LOSSES[args.criterion]
        optimizer = Adam(model.parameters(), lr=args.learning_rate)

        start = time.time()
        for i in (pbar := tqdm(range(1, args.n_iterations + 1))):
            _, loss = train_iteration(
                model,
                optimizer,
                loss_fn,
                *random_training_example(all_categories, category_lines),
            )
            total_loss += loss
            mlflow.log_metric("loss", loss, step=i)

            if i % args.print_every == 0:
                pbar.set_description(
                    f"t={time_since(start)} iter={i:7d} ({i/args.n_iterations*100:4.2f}%) {loss=:6.4f}"
                )

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
