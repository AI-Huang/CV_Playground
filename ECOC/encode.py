#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Jun-20-20 03:43
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org


import os
from itertools import combinations
from generator import hamming_distance


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


def code_set17() -> list:
    """ECOC code set for N=17
    """
    # 硬编码
    code_set = {0: 0b00000000011111111,
                1: 0b00000011100011111,
                2: 0b00000011111100011,
                3: 0b00000101101101101,
                4: 0b00000101110110110,
                5: 0b00000110101111010,
                6: 0b00000110111010101,
                7: 0b00000111010111001,
                8: 0b00000111011001110,
                9: 0b00001001111011001}
    return code_set


def code_set23() -> list:
    """ECOC code set for N=23
    """
    # 硬编码
    code_set = {0: 0b00000000000011111111111,
                1: 0b00000001111100000111111,
                2: 0b00000001111111111000001,
                3: 0b00000110011100111001110,
                4: 0b00000110101111001110010,
                5: 0b00000111110001110110100,
                6: 0b00001011001110110111000,
                7: 0b00001011110011001001110,
                8: 0b00010101010111010011010,
                9: 0b00010101100110101101100}
    return code_set


def min_max_hamming_distance(code_set):
    """一个 code_set 的码字间最小、最大汉明距离
    """
    codes = [_ for _ in code_set.values()]
    min_d, max_d = float("inf"), 0
    for code_pair in combinations(codes, 2):
        d = hamming_distance(code_pair[0], code_pair[1])
        if d < min_d:
            min_d = d
        if d > max_d:
            max_d = d
    return min_d, max_d


def test():
    code_set = code_set23()
    min_d, max_d = min_max_hamming_distance(code_set=code_set)
    print(min_d, max_d)


def test2():
    # 为了加速，直接使用 tensor 数据结构 TODO
    pass


def main():
    test()


if __name__ == "__main__":
    main()
