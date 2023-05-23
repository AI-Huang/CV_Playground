#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Aug-09-22 15:59
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

from torch.utils.data import Dataset


class RMNIST(Dataset):
    def __init__(self, all_data, train=True, transform=None, target_transform=None):
        self.all_data = all_data
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if train:
            self.images = all_data["train_x"]
            self.labels = all_data["train_y"]
        else:
            self.images = all_data["test_x"]
            self.labels = all_data["test_y"]

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
