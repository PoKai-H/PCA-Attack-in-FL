import torch
from torch import nn
import torch.nn.functional as F

class MNISTCNN(nn.Module):
    def __init__(self, use_dropout: bool= False):
        """
        Following the architecture of pytroch official tutorial,
        https://github.com/pytorch/examples/blob/main/mnist/main.py,
        Only defaulting dropout to none to better control the reproductivity
        """
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.dropout1 = nn.Dropout(0.25) if use_dropout else nn.Identity()
        self.dropout2 = nn.Dropout(0.5) if use_dropout else nn.Identity()

        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
