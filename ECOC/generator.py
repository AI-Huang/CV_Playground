#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jun-18-20 20:18
# @Author  : Kelly Hwong (you@example.org)
# @Link    : http://example.org

import time
import numpy as np


def count_ones(n: int):
    count = 0
    while count < n:
        n &= n-1  # 清除最低位的1
        count += 1
    return count


def hamming_distance(x: int, y: int) -> int:
    n = x ^ y
    count = 0
    while count < n:
        n &= n-1  # 清除最低位的1
        count += 1
    return count


def valid_choices(num_choices: int, M: int) -> list:
    choices = []
    for c in range(num_choices):
        if count_ones(c) == M:  # Must contain M 1s
            choices.append(c)
    return choices


def test():
    num_classes = 10
    min_length = np.ceil((np.log2(num_classes))).astype("int")  # 4

    params_table = [(5, 3, 2),
                    (17, 8, 6),
                    (23, 11, 10),
                    (29, 14, 12)]  # (35, 17, N/A)
    N, M, D = params_table[2]  # 0, 1, 2

    start = time.process_time()
    choices = valid_choices(num_choices=2 << N, M=M)
    elapsed = (time.process_time() - start)
    print(f"Time used: {elapsed}s")

    return
    count, code_set = 0, []
    while count < num_classes:
        # Choose one by one 贪婪法
        is_found = False
        for c in choices:  # 顺序查找，结果会与初始选择关系较大
            # Validation
            is_valid = True
            for c2 in code_set:
                if hamming_distance(c, c2) >= D:
                    pass
                else:
                    choices.remove(c)
                    is_valid = False
                    break
            if is_valid:
                # Appending
                code_set.append(c)
                count += 1
                choices.remove(c)
                is_found = True
                break
        if not is_found:
            print("Failed finding all codes.")
            break

    for i, c in enumerate(code_set):
        print(f"Code {i}: {c:029b}")


def test2():
    num_classes = 10
    min_length = np.ceil((np.log2(num_classes))).astype("int")  # 4

    params_table = [(5, 3, 2),
                    (17, 8, 6),
                    (23, 11, 10),
                    (29, 14, 12)]  # (35, 17, N/A)
    N, M, D = params_table[3]  # 0, 1, 2
    num_choices = np.exp2(N).astype("int")
    print(f"num_choices: {num_choices}")  # num_choices=536870912 for N=29
    choices = []
    for c in range(num_choices):
        if count_ones(c) == M:  # Must contain M 1s
            choices.append(c)


def main():
    test()


if __name__ == "__main__":
    main()
