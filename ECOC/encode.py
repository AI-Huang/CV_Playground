#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jun-20-20 03:43
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org


from itertools import combinations
from generator import hamming_distance
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def code_set5() -> list:
    """ECOC code set for N=5
    """
    # 硬编码
    code_set = {0: 0b00111,
                1: 0b01011,
                2: 0b01101,
                3: 0b01110,
                4: 0b10011,
                5: 0b10101,
                6: 0b10110,
                7: 0b11001,
                8: 0b11010,
                9: 0b11100}
    return code_set


def max_hamming_distance(code_set):
    codes = []
    for k in code_set.keys():
        codes.append(code_set[k])
    print(codes)

    max_d = 0
    for code_pair in combinations(codes, 2):
        d = hamming_distance(code_pair[0], code_pair[1])
        if d > max_d:
            max_d = d
    return max_d


def test():
    code_set = code_set5()
    max_d = max_hamming_distance(code_set=code_set)


def test2():
    # 为了加速，直接使用 tensor 数据结构 TODO
    pass


def main():
    test()


if __name__ == "__main__":
    main()
