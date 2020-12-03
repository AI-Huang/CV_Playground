#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Nov-21-20 16:25
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)
# @Link    : http://example.org

from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    """MLP with BatchNorm, ReLU and Dropout
    """

    def __init__(self):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(784, 548)
        self.bc1 = nn.BatchNorm1d(548)

        self.fc2 = nn.Linear(548, 252)
        self.bc2 = nn.BatchNorm1d(252)

        self.fc3 = nn.Linear(252, 10)

    def forward(self, x):
        x = x.view((-1, 784))
        h = self.fc1(x)
        h = self.bc1(h)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)

        h = self.fc2(h)
        h = self.bc2(h)
        h = F.relu(h)
        h = F.dropout(h, p=0.2, training=self.training)

        h = self.fc3(h)
        out = F.log_softmax(h)
        return out


def main():
    pass


if __name__ == "__main__":
    main()
