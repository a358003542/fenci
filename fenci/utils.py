#!/usr/bin/env python
# -*-coding:utf-8-*-

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def normalized_path(path='.') -> str:
    """
    默认支持 ~ 符号

    返回的是字符串

    which default support the `~`
    """
    if isinstance(path, Path):
        return str(path.expanduser())
    elif isinstance(path, str):
        if path.startswith('~'):
            path = os.path.expanduser(path)
        return path
    else:
        raise TypeError


def strdecode(sentence):
    """
    字符串decode 只尝试utf8，和 gbk 否则将抛出异常
    :param sentence:
    :return:
    """
    if not isinstance(sentence, str):
        try:
            sentence = sentence.decode('utf-8')
        except UnicodeDecodeError:
            try:
                sentence = sentence.decode('gbk')
            except UnicodeDecodeError:
                logger.error('I have tried utf8 and gbk encoding, all failed.')
                raise UnicodeDecodeError
    return sentence


try:
    import pkg_resources

    get_module_res = lambda *res: pkg_resources.resource_stream(__name__,
                                                                os.path.join(
                                                                    *res))
except ImportError:
    get_module_res = lambda *res: open(os.path.normpath(os.path.join(
        os.getcwd(), os.path.dirname(__file__), *res)), 'rb')
