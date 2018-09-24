#!/usr/bin/env python
# -*-coding:utf-8-*-

import os
import logging

logger = logging.getLogger(__name__)


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
                                                                os.path.join(*res))
except ImportError:
    get_module_res = lambda *res: open(os.path.normpath(os.path.join(
        os.getcwd(), os.path.dirname(__file__), *res)), 'rb')
