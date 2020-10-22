#!/usr/bin/env python
# -*-coding:utf-8-*-

import re
import math
import threading
import logging
import os
import time
from hashlib import md5
import tempfile
import marshal

from .utils import normalized_path
from . import __softname__
from .hmm import cut as hmm_cut
from .const import DEFAULT_DICT, DEFALUT_CACHE_NAME
from .compat import _replace_file
from .utils import strdecode, get_module_res

DICT_WRITING = {}

logger = logging.getLogger(__name__)

re_userdict = re.compile('^(.+?)( [0-9]+)?( [a-z]+)?$')
re_eng = re.compile('[a-zA-Z0-9]')

re_han_default = re.compile(r"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%\-]+)")
re_skip_default = re.compile(r"([\r\n|\s]+)")


def resolve_filename(f):
    try:
        return f.name
    except AttributeError:
        return repr(f)


default_new_word_find = hmm_cut


class Segment():
    def __init__(self, dictionary=None):
        self.lock = threading.RLock()

        if dictionary is None:
            self.dictionary = DEFAULT_DICT
        else:
            self.dictionary = normalized_path(dictionary)

        self.FREQ = {}
        self.total = 0
        self.user_word_tag_tab = {}
        self.initialized = False
        self.tmp_dir = None
        self.cache_file = None

    def gen_pfdict(self, f):
        lfreq = {}
        ltotal = 0

        f_name = resolve_filename(f)

        for lineno, line in enumerate(f):
            try:
                line = line.strip().decode('utf-8')
                word, freq = line.split(' ')[:2]
                freq = int(freq)
                lfreq[word] = freq
                ltotal += freq

            except ValueError:
                raise ValueError(
                    'invalid dictionary entry in {0} at Line {1}: {2}'.format(
                        f_name, lineno, line))
        f.close()
        return lfreq, ltotal

    def initialize(self, dictionary=None):
        if dictionary:
            abs_path = normalized_path(dictionary)
            if self.dictionary == abs_path and self.initialized:
                return
            else:
                self.dictionary = abs_path
                self.initialized = False
        else:
            abs_path = self.dictionary

        with self.lock:  # 字典在建造时的线程锁
            if self.initialized:  # 已经初始化了就不用初始化了
                return

            logger.debug("Building prefix dict from %s ..." % (
                    abs_path or 'the default dictionary'))
            t1 = time.time()

            if self.cache_file:
                cache_file = self.cache_file
            # default dictionary
            elif abs_path == DEFAULT_DICT:
                cache_file = DEFALUT_CACHE_NAME
            # custom dictionary
            else:
                random_id = md5(abs_path.encode('utf-8', 'replace')).hexdigest()
                cache_file = f"{__softname__}.{random_id}.cache"

            cache_file = os.path.join(self.tmp_dir or tempfile.gettempdir(),
                                      cache_file)
            tmpdir = os.path.dirname(cache_file)

            load_from_cache_fail = True
            if os.path.isfile(cache_file) and (abs_path == DEFAULT_DICT or
                                               os.path.getmtime(
                                                   cache_file) > os.path.getmtime(
                        abs_path)):
                logger.debug("Loading model from cache {0}".format(cache_file))
                try:
                    with open(cache_file, 'rb') as cf:
                        self.FREQ, self.total = marshal.load(cf)
                    load_from_cache_fail = False
                except Exception:
                    load_from_cache_fail = True

            if load_from_cache_fail:
                wlock = DICT_WRITING.get(abs_path, threading.RLock())
                DICT_WRITING[abs_path] = wlock
                with wlock:
                    self.FREQ, self.total = self.gen_pfdict(
                        self.get_dict_file())
                    logger.debug(
                        "Dumping model to file cache {0}".format(cache_file))
                    try:
                        # prevent moving across different filesystems
                        fd, fpath = tempfile.mkstemp(dir=tmpdir)
                        with os.fdopen(fd, 'wb') as temp_cache_file:
                            marshal.dump((self.FREQ, self.total),
                                         temp_cache_file)
                        _replace_file(fpath, cache_file)
                    except Exception:
                        logger.exception("Dump cache file failed.")

                try:
                    del DICT_WRITING[abs_path]
                except KeyError:
                    pass

            self.initialized = True
            logger.debug(
                "Loading model cost %.3f seconds." % (time.time() - t1))
            logger.debug("Prefix dict has been built succesfully.")

    def check_initialized(self):
        if not self.initialized:
            self.initialize()

    def get_DAG(self, sentence):
        self.check_initialized()

        DAG = {}
        N = len(sentence)
        for k in range(N):
            tmplist = []
            i = k
            frag = sentence[k]
            while i < N:
                if self.FREQ.get(frag, 0) > 0:
                    tmplist.append(i)
                i += 1
                frag = sentence[k:i + 1]
            if not tmplist:
                tmplist.append(k)
            DAG[k] = tmplist
        return DAG

    def get_dict_file(self):
        if self.dictionary == DEFAULT_DICT:
            return get_module_res(DEFAULT_DICT)
        else:
            return open(self.dictionary, 'rb')

    def calc(self, sentence, DAG, route):
        N = len(sentence)
        route[N] = (0, 0)

        logtotal = math.log(self.total)
        for idx in range(N - 1, -1, -1):  # 逆序规划 选择一条整个路径频率最大的句子
            route[idx] = max(
                (math.log(self.FREQ.get(sentence[idx:x + 1]) or 1) -
                 logtotal + route[x + 1][0],
                 x) for x in DAG[idx])  # x 终点索引点 idx 考察开始点

    def __cut_DAG(self, sentence, new_word_find=None):

        if new_word_find is None:
            new_word_find = default_new_word_find

        DAG = self.get_DAG(sentence)
        route = {}
        self.calc(sentence, DAG, route)

        x = 0
        buf = ''
        N = len(sentence)
        while x < N:
            y = route[x][1] + 1
            l_word = sentence[x:y]
            if y - x == 1:
                buf += l_word  # 单字母或单字
            else:
                if buf:
                    if len(buf) == 1:  # 夹着的单字
                        yield buf
                        buf = ''
                    else:
                        if not self.FREQ.get(buf):  # 词典里找不到的词 用HMM来分
                            recognized = new_word_find(buf)
                            for t in recognized:
                                yield t
                        else:
                            for elem in buf:
                                yield elem
                        buf = ''

                yield l_word  # 找到的词优先输出
            x = y

        # 纯单字母或单字的情况
        if buf:
            if len(buf) == 1:
                yield buf
            elif not self.FREQ.get(buf):
                recognized = hmm_cut(buf)
                for t in recognized:
                    yield t
            else:
                for elem in buf:
                    yield elem

    def cut(self, sentence):

        """
        The main function that segments an entire sentence that contains
        Chinese characters into seperated words.

        Parameter:
            - sentence: The str(unicode) to be segmented.
            - HMM: Whether to use the Hidden Markov Model.
        """
        sentence = strdecode(sentence)

        re_han = re_han_default
        re_skip = re_skip_default

        cut_block = self.__cut_DAG

        blocks = re_han.split(sentence)

        for blk in blocks:

            if not blk:  # 空白符号跳过
                continue
            if re_han.match(blk):  # 中文符号 核心分词在这里
                for word in cut_block(blk):
                    yield word
            else:
                tmp = re_skip.split(blk)
                for x in tmp:

                    if re_skip.match(x):  # 多个空白不分开
                        yield x
                    else:
                        for xx in x:  # 剩下来的全部分开
                            yield xx

    def lcut(self, sentence):
        return list(self.cut(sentence))
