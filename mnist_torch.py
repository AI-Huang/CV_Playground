#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Sep-22-20 20:46
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)
# @RefLink : https://github.com/pytorch/examples/blob/master/mnist/main.py


from __future__ import print_function
import argparse
from datetime import datetime
import json
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from models.torch_fn.lenet import LeNet5


def cmd_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model-name', type=str, default="LeNet-5")
    parser.add_argument('--dataset-name', type=str, default="MNIST")
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')

    parser.add_argument('--do-train', action='store_true', default=True)
    parser.add_argument('--do-eval', action='store_true', default=True)

    parser.add_argument('--loss', type=str, default='cross_entropy')
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    parser.add_argument('--gamma', type=float, default=0.7,
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--no-norm', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    return args


def train(model, train_loader, criterion, optimizer, epoch, args):
    model.train()
    device = args.device
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, test_loader, args):
    model.eval()
    device = args.device
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset), test_accuracy)
    )

    return {"test_loss": test_loss, "test_accuracy": test_accuracy}


def main():
    args = cmd_args()
    date_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(
        "output", f"{args.model_name}_{args.dataset_name}_{args.optimizer}_{date_time}")
    os.makedirs(output_dir, exist_ok=True)

    # Record config
    with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf8') as json_file:
        json_file.write(json.dumps(vars(args)))
    # Log test results, write head row
    with open(os.path.join(output_dir, 'test.csv'), "a") as f:
        f.write(",".join(["epoch", "test_loss", "test_accuracy"])+'\n')

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    args.device = device
    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                      )

    if not args.no_norm:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        print("No normalization")
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

    mnist_train = datasets.MNIST(os.path.expanduser(
        "~/.datasets"), train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(os.path.expanduser(
        "~/.datasets"), train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(mnist_train, **kwargs)
    test_loader = torch.utils.data.DataLoader(mnist_test, **kwargs)

    model = LeNet5(output_dim=10).to(device)
    if args.loss == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    scheduler = None
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr, momentum=args.momentum)
    else:
        # TODO, other optimizers such as Adam
        StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in tqdm(range(args.epochs)):
        if args.do_train:
            train(model, train_loader, criterion, optimizer, epoch, args)
        if args.do_eval:
            test_result = test(model, test_loader, args)
            with open(os.path.join(output_dir, 'test.csv'), "a") as f:
                f.write(",".join(
                    [str(_) for _ in [
                        epoch, test_result["test_loss"], test_result["test_accuracy"]
                    ]]
                ) + '\n')

        if scheduler:
            scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(),
                   os.path.join(output_dir, f"{args.model_name}.pt"))


if __name__ == '__main__':
    main()
