# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

#obvi just use the actual NN but for the structure
class SimpleModel(nn.Module):
    def __init__(self, input_size=2, output_size=4):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # action scores