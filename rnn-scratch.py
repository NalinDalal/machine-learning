"""
01_basic_rnn_cell_from_scratch.py

Implements a single-layer vanilla RNN manually using PyTorch tensors.
Equation:
    h_t = tanh(x_t @ W_x + h_{t-1} @ W_h + b)

No nn.RNN used — pure matrix math.
"""

import torch
import torch.nn as nn

torch.manual_seed(42)

# Hyperparameters
input_size = 3
hidden_size = 5
seq_len = 4
batch_size = 2

# Random input: [batch, seq_len, input_size]
X = torch.randn(batch_size, seq_len, input_size)

# Parameters
W_x = nn.Parameter(torch.randn(input_size, hidden_size))
W_h = nn.Parameter(torch.randn(hidden_size, hidden_size))
b = nn.Parameter(torch.zeros(hidden_size))

def forward(X):
    h = torch.zeros(batch_size, hidden_size)
    outputs = []

    for t in range(seq_len):
        x_t = X[:, t, :]
        h = torch.tanh(x_t @ W_x + h @ W_h + b)
        outputs.append(h)

    return torch.stack(outputs, dim=1)

Y = forward(X)
print("Output shape:", Y.shape)
