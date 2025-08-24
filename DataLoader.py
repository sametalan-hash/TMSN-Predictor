import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def load_data(batch_size=1):
    # Assuming you have your stock data in numpy arrays
    # features: (num_samples, sequence_length, num_features)
    # targets: (num_samples, 1)
    
    features = np.random.randn(1000, 10, 12)  # Example data
    targets = np.random.randn(1000, 1)

    # Convert to torch tensors
    features = torch.tensor(features, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)

    dataset = TensorDataset(features, targets)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=batch_size)

    return train_loader, test_loader