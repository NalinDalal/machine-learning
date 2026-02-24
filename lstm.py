"""
04_lstm_sequence_generator.py

Train LSTM on sine wave and generate sequence.
"""

import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        return self.fc(out), hidden

model = LSTMGenerator().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
for step in range(2000):
    X_batch, y_batch = generate_data()
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

    outputs, _ = model(X_batch)
    loss = criterion(outputs, y_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Generation
model.eval()
sequence = [0.] * 20

for _ in range(100):
    X = torch.tensor(sequence[-20:], dtype=torch.float32).view(1, 20, 1).to(device)
    with torch.no_grad():
        y_pred, _ = model(X)
    next_val = y_pred[0, -1, 0].item()
    sequence.append(next_val)

print("Generated sequence length:", len(sequence))
