#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Apr-17-20 02:45
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import json
import pandas as pd
import numpy as np

import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from model import LeNet5
from utils import makedir_exist_ok

# parameters
BATCH_SIZE = 64


def main():
    print("Loading config...")
    with open('./config.json', 'r') as f:
        CONFIG = json.load(f)
    DATASETS_DIR = CONFIG["DATASETS_DIR"]
    MODEL_DIR = CONFIG["MODEL_DIR"]
    makedir_exist_ok(MODEL_DIR)

    print("Step 1: Preparing data...")

    # print("Step 2: Converting data...")
    # X = data_train[:, 1:].reshape(data_train.shape[0], 1, 28, 28)
    # X = X.astype(float)
    # y = data_train[:, 0]
    # y = y.astype(int)

    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.ToTensor()
        ]),
    }

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    data_train = datasets.MNIST(
        root=DATASETS_DIR, train=True, download=True)  # transform=transform
    data_train = datasets.MNIST(
        root=DATASETS_DIR, train=True, transform=transform, download=True)
    data_test = datasets.MNIST(
        root=DATASETS_DIR, train=False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(dataset=data_train,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset=data_test,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=4)

    print("Step 2: Training config...")
    num_epoch = 1  # 1000
    torch.manual_seed(42)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("CUDA GPU available!")
    device = torch.device("cuda" if use_cuda else "cpu")

    print("Step 3: Training phase...")
    num_train = len(data_train)
    nb_index = 0
    batch_size = BATCH_SIZE

    model = LeNet5().to(device)
    model.train()

    criterion = nn.CrossEntropyLoss(size_average=False)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, betas=(0.9, 0.99))

    for epoch in range(num_epoch):  # a total iteration/epoch
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            if use_cuda:
                X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
            X_batch, y_batch = Variable(X_batch), Variable(y_batch)
            print(y_batch)
            print(y_batch.type)
            input()
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:  # print every 100 steps
                print(
                    f"Train epoch: {epoch}, [{batch_idx*batch_size}/{num_train} ({batch_idx*batch_size/num_train*100:.2f}%)].\tLoss: {loss:.6f}")

    model_path = os.path.join(MODEL_DIR, "test.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}.")


if __name__ == "__main__":
    main()
