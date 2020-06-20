#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jun-20-20 15:04
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import time
import numpy as np
from itertools import combinations


def valid_choices_combinations(N, M):
    bit_positions = [_ for _ in range(N)]  # 0~(N-1)

    start = time.process_time()
    combs = []
    for comb in combinations(bit_positions, M):  # choose positions for M 1s
        combs.append(comb)
    print(len(combs))  # 77558760
    elapsed = (time.process_time() - start)
    print(f"Time used: {elapsed}s")  # N=29, 17.9375s

    start = time.process_time()
    choices = []
    for i, comb in enumerate(combs):
        choice = 0
        for pos in comb:
            choice += 1 << pos  # instead of using 2**pos
        choices.append(choice)
        # if i > 10_0000:  # N=29, 0.171875s for 10_1000, will use 133.3s for all 77558760 combs
        # break
    elapsed = (time.process_time() - start)
    print(f"Time used: {elapsed}s")
    return choices


def main():
    """N长的二进制数里寻找所有1的个数是M的数
    """
    num_classes = 10
    min_length = np.ceil((np.log2(num_classes))).astype("int")  # 4

    params_table = [(5, 3, 2),
                    (17, 8, 6),
                    (23, 11, 10),
                    (29, 14, 12)]  # (35, 17, N/A)
    N, M, D = params_table[2]  # 0, 1, 2, 3(算不完)
    num_choices = 2 << N

    start = time.process_time()
    choices = valid_choices_combinations(N, M)  # N=23, 2.109375s
    elapsed = (time.process_time() - start)
    print(f"Time used: {elapsed}s")


if __name__ == "__main__":
    main()
