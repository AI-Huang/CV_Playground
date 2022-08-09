#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Apr-17-20 02:55
# @Author  : Kan HUANG (kanhuang@astri.org)
# @RefLink : https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """LeNet5 implemented with PyTorch
    LeNet5 的结构，是 3x conv2d 和 2x FC
    Inputs:
        input_shape: input shape of the element of the batched data, e.g., (32, 32, 3), (28, 28, 1).
        output_dim: number of top classifiers, e.g., 2, 10.
        attention: TODO, attention type, one of["official", "senet"], default None.
    """

    def __init__(self, output_dim=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)  # padding 28*28 to 32*32
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_dim)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # conv1 -> max_pool
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))  # conv2 -> max_pool
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class LeNet5_RGB(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
