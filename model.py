import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_categories):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(n_categories + input_size, hidden_size, batch_first=True)
        self.o2o = nn.Linear(hidden_size + input_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=2)
        self.num_directions = 1

    def forward(self, category, input_tensor, hidden):

        input_combined = torch.cat(
            (category, input_tensor), dim=2
        )  # Concatenate along the feature dimension
        output, hidden = self.rnn(input_combined, hidden)
        output_combined = torch.cat((output, input_tensor), dim=2)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros(self.num_directions, batch_size, self.hidden_size)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_categories):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(n_categories + input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, category, input_tensor, hidden):

        input_combined = torch.cat(
            (category, input_tensor), dim=2
        )  # Concatenate along the feature dimension
        output, (hidden, cell) = self.lstm(input_combined, hidden)
        output = self.fc(output)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, (hidden, cell)

    def init_hidden(self, batch_size=1):
        return (
            torch.zeros(1, batch_size, self.hidden_size),
            torch.zeros(1, batch_size, self.hidden_size),
        )  # (hidden_state, cell_state)


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_categories):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(n_categories + input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=2)
        self.num_directions = 1

    def forward(self, category, input_tensor, hidden):

        input_combined = torch.cat(
            (category, input_tensor), dim=2
        )  # Concatenate along the feature dimension
        output, hidden = self.gru(input_combined, hidden)
        output = self.fc(output)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros(
            self.num_directions, batch_size, self.hidden_size
        )  # GRU only needs hidden state, not cell state


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        n_categories,
        num_heads=4,
        num_layers=2,
    ):
        super(TransformerModel, self).__init__()
        self.hidden_size = hidden_size

        self.letter_embedding = nn.Embedding(input_size, hidden_size)
        self.category_embedding = nn.Embedding(n_categories, hidden_size)
        self.positional_embedding = nn.Embedding(100, hidden_size)  # 100 = max length
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads
        )

        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, category, input_tensor, hidden=None):

        input_combined = torch.cat(
            (category, input_tensor), dim=2
        )  # Concatenate along the feature dimension

        # get category, letter and positional indices
        letter_indices = torch.argmax(input_tensor, dim=2)
        category_indices = torch.argmax(category, dim=2)
        positional_indices = torch.arange(input_combined.size(1)).expand(
            input_combined.size(0), input_combined.size(1)
        )

        # compute embeddings and sum
        letter_embeddings = self.letter_embedding(letter_indices)
        category_embeddings = self.category_embedding(category_indices)
        positional_embeddings = self.positional_embedding(positional_indices)
        embeddings = letter_embeddings + category_embeddings + positional_embeddings

        # transformer layer
        output = self.transformer(embeddings)

        # finaL output layer
        output = self.fc(output)
        output = self.softmax(output)
        return output, None

    def init_hidden(self, batch_size: int = 1):
        return None  # Transformer doesn't use hidden states


class MixtureOfExperts(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, n_categories, num_experts=3
    ):
        super(MixtureOfExperts, self).__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts

        self.experts = nn.ModuleList(
            [
                nn.Linear(n_categories + input_size, hidden_size)
                for _ in range(num_experts)
            ]
        )
        self.gate = nn.Linear(n_categories + input_size, num_experts)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, category, input_tensor, hidden):
        input_combined = torch.cat(
            (category, input_tensor), dim=2
        )  # Concatenate along the feature dimension

        # Compute gating scores
        gate_scores = torch.softmax(self.gate(input_combined), dim=1)

        # Compute expert outputs
        expert_outputs = torch.stack(
            [expert(input_combined) for expert in self.experts], dim=2
        )

        # Weighted sum of expert outputs
        output = torch.sum(gate_scores.unsqueeze(3) * expert_outputs, dim=2)
        output = self.fc(output)
        output = self.softmax(output)

        return output, None  # No explicit hidden state

    def init_hidden(self, batch_size: int = 1):
        return None  # Not needed
