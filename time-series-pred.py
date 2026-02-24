"""
03_time_series_prediction.py

RNN predicting next value in sine wave.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate synthetic sine data
def generate_data(seq_len=20, batch_size=32):
    X = []
    y = []
    for _ in range(batch_size):
        start = np.random.rand() * 2 * np.pi
        seq = np.sin(np.linspace(start, start + seq_len + 1, seq_len + 1))
        X.append(seq[:-1])
        y.append(seq[1:])
    return torch.tensor(X, dtype=torch.float32).unsqueeze(-1), \
           torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

# Model
class RNNRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(1, 50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out)

model = RNNRegressor().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
for step in range(2000):
    X_batch, y_batch = generate_data()
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

    outputs = model(X_batch)
    loss = criterion(outputs, y_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(f"Step {step}, MSE: {loss.item():.6f}")
