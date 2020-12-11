#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Aug-06-20 14:33
# @Author  : Kelly Hwong (dianhuangkan@gmail.com)

import errno
import six
import os
import pathlib
import logging


_logger = logging.getLogger(__name__)


def create_dir(dir_path):
    """
    Create directory if it does not exist
    """
    try:
        if not os.path.exists(dir_path):
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)

        return 0
    except Exception as err:
        _logger.critical('Creating directories error: {0}'.format(err))
        exit(-1)


# See somewhere
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

# from tensorflow.python.keras.utils import data_utils


def _makedirs_exist_ok(datadir):
    """makedir_exist_ok compatible for both Python 2 and Python 3
    """
    if six.PY3:
        os.makedirs(
            datadir, exist_ok=True)  # pylint: disable=unexpected-keyword-arg
    else:
        # Python 2 doesn't have the exist_ok arg, so we try-except here.
        try:
            os.makedirs(datadir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
