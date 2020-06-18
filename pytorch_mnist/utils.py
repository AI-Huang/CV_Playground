#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Apr-17-20 16:36
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import os
import errno


def makedir_exist_ok(dirpath):
    """
    Python2 support for os.makedirs(.., exist_ok=True)
    """
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise


def test():
    dirpath = "D:\DeepLearningData\pytorch_mnist\models"
    makedir_exist_ok(dirpath)


def main():
    test()


if __name__ == "__main__":
    main()
