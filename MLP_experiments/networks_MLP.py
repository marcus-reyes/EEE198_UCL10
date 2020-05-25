# MLP models for MNIST
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MaskedLinear(nn.Linear):
    """Custom layer to allow layer masking """

    def __init__(self, in_features, out_features, bias=False):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.mask = torch.ones_like(self.weight.data, device=DEVICE)

    def forward(self, x):
        # main modification, weight is multiplied with the mask
        masked_weight = torch.mul(self.weight, self.mask)
        return F.linear(x, masked_weight, self.bias)


class Net(nn.Module):
    """Normal 4-layer MLP network"""

    def __init__(self):
        super(Net, self).__init__()

        # define layers
        self.fc1 = MaskedLinear(28 ** 2, 512)
        self.fc2 = MaskedLinear(512, 256)
        self.fc3 = MaskedLinear(256, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)  # flatten spatial dim
        # self.fc1.weight = torch.nn.Parameter(self.fc1.weight*self.mask1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x


class SimplestNet(nn.Module):
    """Simplest possible network for MNIST"""

    def __init__(self):
        super(SimplestNet, self).__init__()

        # 1 layer only
        self.fc1 = MaskedLinear(28 ** 2, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)  # flatten spatial dim
        x = F.log_softmax(self.fc1(x), dim=1)
        return x


class SimpleNet(nn.Module):
    """Simple 2-layer MLP"""
    def __init__(self):
        super(SimpleNet, self).__init__()

        # define layers
        self.fc1 = MaskedLinear(28 ** 2, 256)
        self.fc4 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)  # flatten spatial dim
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x


class DeconsNet(nn.Module):
    """Similar FC from Deconstructing LTs"""

    def __init__(self):
        super(DeconsNet, self).__init__()

        # define layers
        self.fc1 = MaskedLinear(28 ** 2, 300)
        self.fc2 = MaskedLinear(300, 100)
        self.fc3 = MaskedLinear(100, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)  # flatten spatial dim
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x
