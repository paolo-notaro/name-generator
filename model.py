import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_categories):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input_tensor, hidden):
        input_combined = torch.cat((category, input_tensor, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_categories):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(n_categories + input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input_tensor, hidden):
        input_combined = torch.cat((category, input_tensor), dim=1).unsqueeze(
            1
        )  # Add sequence dimension
        output, (hidden, cell) = self.lstm(input_combined, hidden)
        output = self.fc(output.squeeze(1))
        output = self.dropout(output)
        output = self.softmax(output)
        return output, (hidden, cell)

    def init_hidden(self):
        return (
            torch.zeros(1, 1, self.hidden_size),
            torch.zeros(1, 1, self.hidden_size),
        )  # (hidden_state, cell_state)


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_categories):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(n_categories + input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input_tensor, hidden):
        input_combined = torch.cat((category, input_tensor), dim=1).unsqueeze(
            1
        )  # Add sequence dimension
        output, hidden = self.gru(input_combined, hidden)
        output = self.fc(output.squeeze(1))
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(
            1, 1, self.hidden_size
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

        self.embedding = nn.Linear(n_categories + input_size, hidden_size)
        self.pos_encoder = nn.Embedding(10, hidden_size)  # Simple positional encoding
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input_tensor, hidden=None):
        input_combined = torch.cat((category, input_tensor), dim=1).unsqueeze(
            0
        )  # (seq_len=1, batch=1, feature_dim)
        embedded = self.embedding(input_combined)
        position_ids = torch.arange(1).unsqueeze(0)  # Fake position ids
        pos_encoded = embedded + self.pos_encoder(position_ids)
        output = self.transformer(pos_encoded)
        output = self.fc(output.squeeze(0))
        output = self.softmax(output)
        return output, None  # Transformers do not maintain hidden states

    def init_hidden(self):
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
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input_tensor, hidden):
        input_combined = torch.cat((category, input_tensor), dim=1)

        # Compute gating scores
        gate_scores = torch.softmax(self.gate(input_combined), dim=1)

        # Compute expert outputs
        expert_outputs = torch.stack(
            [expert(input_combined) for expert in self.experts], dim=1
        )

        # Weighted sum of expert outputs
        output = torch.sum(gate_scores.unsqueeze(2) * expert_outputs, dim=1)
        output = self.fc(output)
        output = self.softmax(output)

        return output, None  # No explicit hidden state

    def init_hidden(self):
        return None  # Not needed
