#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jun-20-20 22:14
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import torch
import numpy as np
from encode import code_set5


def _test():
    use_cuda = torch.cuda.is_available()

    # get batch tensor
    a = torch.Tensor([0, 7, 5, 6, 2, 1, 7, 2, 9, 4, 0, 8, 5, 8, 3, 3, 5, 2, 5, 5, 7, 7, 6, 5, 7, 7, 6, 2, 9,
                      3, 0, 0, 7, 7, 3, 7, 8, 7, 8, 0, 2, 8, 2, 3, 8, 1, 8, 2, 5, 6, 2, 1, 5, 7, 9, 5, 1, 5, 3, 4, 1, 4, 9, 7])
    print(a.size())  # torch.Size([64])

    code_set = code_set5()  # in CPU, dict: int->int

    # int(): map using CPU then to Tensor
    decimal = torch.Tensor([code_set[int(_)] for _ in a])  # 64 * 1
    N = 5

    print(decimal)
    input()
    b = torch.zeros(64, 5)
    for i, _ in enumerate(decimal.clone()):
        bits = torch.zeros(N)
        print(a[i])
        for j in range(N):
            if _ % 2 == 1:
                bits[j] = 1
            _ //= 2  # floor div
        print(bits)
        input()
        b[i] = bits.clone()

    print(b)
    # to 64 * 5|num_bits
    if use_cuda:
        a = a.cuda()

    print(a)


if __name__ == "__main__":
    _test()
