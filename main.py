import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# ===============================
# 1. Transformer Model Definition
# ===============================
class TransformerModel(nn.Module):
    def __init__(self, feature_size=10, embed_size=256, num_heads=8, num_layers=4, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(feature_size, embed_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True   # ✅ Fixes the warning
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(embed_size, 1)

    def forward(self, x):
        # x: (batch_size, seq_len, feature_size)
        x = self.embedding(x)  # (batch_size, seq_len, embed_size)
        x = self.transformer_encoder(x)  # (batch_size, seq_len, embed_size)
        out = self.fc_out(x[:, -1, :])  # use last timestep
        return out


# ===============================
# 2. Synthetic Dataset (replace with stock data later)
# ===============================
def generate_dummy_data(n_samples=1000, seq_len=20, feature_size=10):
    X = np.random.rand(n_samples, seq_len, feature_size).astype(np.float32)
    y = np.sum(X[:, -1, :], axis=1, keepdims=True)  # simple target
    return torch.tensor(X), torch.tensor(y)


# ===============================
# 3. Train & Evaluate
# ===============================
def train_model(model, train_loader, criterion, optimizer, epochs=20):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")


def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            loss = criterion(output, y_batch)
            test_loss += loss.item()
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.6f}")


# ===============================
# 4. Main
# ===============================
def main():
    # Hyperparameters
    feature_size = 10
    seq_len = 20
    batch_size = 32
    embed_size = 256
    num_heads = 8
    num_layers = 4
    dropout = 0.1
    learning_rate = 1e-3
    epochs = 20

    # Data
    X, y = generate_dummy_data(1000, seq_len, feature_size)
    train_size = int(0.8 * len(X))
    test_size = len(X) - train_size
    X_train, X_test = torch.split(X, [train_size, test_size])
    y_train, y_test = torch.split(y, [train_size, test_size])

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    model = TransformerModel(feature_size, embed_size, num_heads, num_layers, dropout)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train & Evaluate
    train_model(model, train_loader, criterion, optimizer, epochs)
    evaluate_model(model, test_loader, criterion)


if __name__ == "__main__":
    main()
