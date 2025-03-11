# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 14:08:22 2025

@author: MIALON alexis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Check GPU availability
print("CUDA Available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(20, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

model = SimpleModel().to(device)
print(model)  # Print model summary-like info

# Generate dummy data
data = np.random.rand(10, 20).astype(np.float32)  # 10 samples, 20 features each
data_torch = torch.from_numpy(data).to(device)

# Forward pass
with torch.no_grad():  # Disable gradient calculations for this test
    predictions = model(data_torch)

print("Predictions:", predictions)