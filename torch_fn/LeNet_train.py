#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Apr-17-20 02:45
# @Author  : Kelly Hwong (you@example.org)
# @Link    : https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import os
import json
import pandas as pd
import numpy as np
import csv

import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from LeNet import LeNet5
from utils import makedir_exist_ok

# parameters
BATCH_SIZE = 32  # 64 too large

print("Loading config...")
with open('./config.json', 'r') as f:
    CONFIG = json.load(f)
DATASETS_DIR = CONFIG["DATASETS_DIR"]
MODEL_DIR = CONFIG["MODEL_DIR"]
makedir_exist_ok(MODEL_DIR)


def train(model, dataset, epochs=50, batch_size=32, use_cuda=False, seed=None):
    """training process, use a model to train data
    Inputs:
        model: model.
        dataset: dataset to be trained.
        epochs: training epochs.
        seed:
    """
    if seed:  # set RNG seed
        torch.manual_seed(seed)

    model.train()

    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               shuffle=True, num_workers=4)
    num_train = len(dataset)

    criterion = nn.CrossEntropyLoss(size_average=False)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, betas=(0.9, 0.99))

    headers = ["epoch", 'loss']
    f = open("./logs/log_torch_mnist.csv", 'w', newline='')
    f_csv = csv.writer(f)
    f_csv.writerow(headers)

    for epoch in range(epochs):  # a total iteration/epoch
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            if use_cuda:  # config device
                X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:  # print every 100 steps
                print(
                    f"Train epoch: {epoch}, [{batch_idx*batch_size}/{num_train} ({batch_idx*batch_size/num_train*100:.2f}%)].\tLoss: {loss:.6f}")
                f_csv.writerow([epoch, float(loss)])


def test(data_test):
    test_loader = torch.utils.data.DataLoader(dataset=data_test,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=4)

    print("Step 2: Testing config...")
    num_epoch = 1  # 1000
    torch.manual_seed(42)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("CUDA GPU available!")
    device = torch.device("cuda" if use_cuda else "cpu")

    print("Step 3: Testing phase...")
    num_test = len(test_loader.dataset)
    batch_size = BATCH_SIZE

    model = LeNet5().to(device)
    model_path = os.path.join(MODEL_DIR, "test.pth")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    criterion = nn.CrossEntropyLoss(size_average=False)

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (X_batch, y_batch) in enumerate(test_loader):
            if use_cuda:
                X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
            X_batch, y_batch = Variable(X_batch), Variable(y_batch)
            output = model(X_batch)
            test_loss += criterion(output, y_batch)
            # predict by argmax
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(y_batch.view_as(pred)).sum()

    test_loss /= num_test
    acc = 100.0 * correct / num_test
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{num_test} ({acc:.2f}%).\n")


def main():
    # Preparing data
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

    data_train_no_transform = datasets.MNIST(
        root=DATASETS_DIR, train=True, download=True)
    data_train = datasets.MNIST(
        root=DATASETS_DIR, train=True, transform=transform, download=True)
    data_test = datasets.MNIST(
        root=DATASETS_DIR, train=False, transform=transform, download=True)

    # reproducibility
    seed = 42  # seed set to 42

    # Preparing model
    use_cuda = torch.cuda.is_available()  # CUDA
    if use_cuda:
        print("CUDA GPU available! Use GPU.")
    device = torch.device("cuda" if use_cuda else "cpu")

    print(f"Set random seed to {seed} for model.")
    torch.manual_seed(seed)
    model = LeNet5().to(device)

    import time
    start = time.process_time()
    # print(f"Set random seed to {seed} for training.")
    train(model, dataset=data_train, epochs=1,
          batch_size=32, use_cuda=use_cuda, seed=None)  # do not refresh the seed for training
    elapsed = time.process_time() - start
    print(f"Time used: {elapsed}s")

    model_path = os.path.join(MODEL_DIR, "test.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}.")

    # test(data_test)


if __name__ == "__main__":
    main()
