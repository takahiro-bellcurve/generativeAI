import os

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from auto_encoder import AutoEncoder


np.random.seed(1234)
torch.manual_seed(1234)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# データの取得
root = os.path.join('data', 'mnist')
transform = transforms.Compose([transforms.ToTensor(),
                                lambda x: x.view(-1)])
mnist_train = \
    torchvision.datasets.MNIST(root=root,
                               download=True,
                               train=True,
                               transform=transform)
mnist_test = \
    torchvision.datasets.MNIST(root=root,
                               download=True,
                               train=False,
                               transform=transform)
train_dataloader = DataLoader(mnist_train,
                              batch_size=100,
                              shuffle=True)
test_dataloader = DataLoader(mnist_test,
                             batch_size=1,
                             shuffle=False)

# set model
model = AutoEncoder().to('cuda:0')

criterion = nn.BCELoss()

# set optimizer
optimizer = torch.optim.Adam(model.parameters())

epochs = 10

for epoch in range(epochs):
    train_loss = 0.0

    for (x, _) in train_dataloader:

        x = x.to(device)
        y = model(x)
        loss = criterion(y, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_dataloader)

    print(f'epoch: {epoch+1}, loss: {train_loss:.4f}')
