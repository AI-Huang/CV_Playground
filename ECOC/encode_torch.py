#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jun-20-20 22:14
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import torch
import numpy as np
from encode import code_set5


def main():
    a = torch.Tensor([0, 7, 5, 6, 2, 1, 7, 2, 9, 4, 0, 8, 5, 8, 3, 3, 5, 2, 5, 5, 7, 7, 6, 5, 7, 7, 6, 2, 9,
                      3, 0, 0, 7, 7, 3, 7, 8, 7, 8, 0, 2, 8, 2, 3, 8, 1, 8, 2, 5, 6, 2, 1, 5, 7, 9, 5, 1, 5, 3, 4, 1, 4, 9, 7])
    print(a.size())  # torch.Size([64])

    code_set = code_set5()  # dict: int->int
    # map
    a = torch.Tensor([code_set[int(_)] for _ in a])
    print(a)


if __name__ == "__main__":
    main()
