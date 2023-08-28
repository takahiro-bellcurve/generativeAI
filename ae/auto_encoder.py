import torch
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, device='cuda:0'):
        super().__init__()
        self.device = device
        self.l1 = nn.Linear(784, 200)  # 28*28*1 = 784
        self.l2 = nn.Linear(200, 784)

    def forward(self, x):
        # encoding
        h = self.l1(x)

        # activation function
        h = torch.relu(h)

        # decoding
        h = self.l2(h)

        y = torch.sigmoid(h)

        return y
