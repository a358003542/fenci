#!/usr/bin/env python
# -*-coding:utf-8-*-

import re
import math
import threading
import logging
import os
import time

from simple_nltk import FreqDist
from simple_nltk.tokenize.api import TokenizerI

from .base import BaseSegment
from .hmm_segment import HMMSegment
from .utils import normalized_path, get_json_value, set_json_value
from . import __softname__
from .const import DEFAULT_DICT, DEFALUT_CACHE_NAME, CACHE_WRITING
from .utils import strdecode

logger = logging.getLogger(__name__)

re_userdict = re.compile('^(.+?)( [0-9]+)?( [a-z]+)?$')
re_eng = re.compile('[a-zA-Z0-9]')

re_han_default = re.compile(r"([\u4E00-\u9FD5a-zA-Z0-9+#&\._%\-]+)")
re_skip_default = re.compile(r"([\r\n|\s]+)")


class Segment(TokenizerI, BaseSegment):
    def __init__(self, dictionary=None, traning_root=None,
                 traning_regexp='.*\.txt'):
        self.lock = threading.RLock()

        self.training_root = traning_root
        self.training_regexp = traning_regexp

        if dictionary is None:
            self.dictionary = DEFAULT_DICT
            self.dictionary_type = 'default'
        else:
            self.dictionary = normalized_path(dictionary)
            self.dictionary_type = 'custom'

        self.word_fd = FreqDist()

        self.cache_file = DEFALUT_CACHE_NAME

        self.hmm_segment = HMMSegment(traning_root=traning_root,
                                      traning_regexp=traning_regexp,
                                      cache_file=self.cache_file)

        self.initialized = False
        self.tmp_dir = None

    def training(self, root=None, regexp=None):
        """
        根据已经分好词的内容来训练
        :param root:
        :param regexp:
        :return:
        """
        self.check_initialized()

        if root is None and self.training_root is None:
            raise Exception('please give the training data root')

        root = root if root is not None else self.training_root
        regexp = regexp if regexp is not None else self.training_regexp

        from fenci.utils import read_training_content
        content = read_training_content(root, regexp)

        content_list = content.split()

        fd = FreqDist(content_list)

        self.word_fd.update(fd)

    def training_hmm(self, root=None, regexp=None, update_dict=False):
        self.check_initialized()

        if root is None and self.training_root is None:
            raise Exception('please give the training data root')
        root = root if root is not None else self.training_root
        regexp = regexp if regexp is not None else self.training_regexp

        if update_dict:
            self.training(root, regexp)

        self.hmm_segment.training(root, regexp)

    def gen_word_fd(self, filename):
        word_fd = FreqDist()

        with open(filename, 'rt', encoding='utf8') as f:
            for line in f:
                word, freq = line.split()[:2]
                freq = int(freq)
                word_fd.update({word: freq})

        return word_fd

    def initialize(self):
        with self.lock:  # 字典在建造时的线程锁
            if self.initialized:  # 已经初始化了就不用初始化了
                return

            logger.debug("Building prefix dict from %s ..." % (
                    self.dictionary or 'the default dictionary'))
            t1 = time.time()

            cache_file = self._get_cache_file()

            # use cache data
            use_cache_data = False
            if os.path.isfile(cache_file):
                word_fd_timestamp = get_json_value(cache_file,
                                                   'word_fd_timestamp')
                if word_fd_timestamp:
                    if self.dictionary_type == 'custom':
                        if int(word_fd_timestamp) > os.path.getmtime(
                                self.dictionary):
                            use_cache_data = True
                    else:
                        use_cache_data = True

            if use_cache_data:
                logger.debug("Loading model from cache {0}".format(cache_file))

                word_fd = get_json_value(cache_file, 'word_fd')
                self.word_fd = FreqDist(word_fd)
            else:
                word_fd = self.gen_word_fd(self._get_dict_file())
                self.word_fd = FreqDist(word_fd)

                self.save_model(save_hmm=False)

            self.initialized = True
            logger.debug(
                "Loading model cost %.3f seconds." % (time.time() - t1))
            logger.debug("Prefix dict has been built succesfully.")

    def _get_dict_file(self):
        if self.dictionary == DEFAULT_DICT:
            from pkg_resources import resource_filename
            return resource_filename(__softname__, self.dictionary)
        else:
            return self.dictionary

    def get_DAG(self, sentence):
        self.check_initialized()

        DAG = {}
        N = len(sentence)
        for k in range(N):
            tmplist = []
            i = k
            frag = sentence[k]
            while i < N:
                if self.word_fd.get(frag, 0) > 0:
                    tmplist.append(i)
                i += 1
                frag = sentence[k:i + 1]
            if not tmplist:
                tmplist.append(k)
            DAG[k] = tmplist
        return DAG

    def calc(self, sentence, DAG, route):
        N = len(sentence)
        route[N] = (0, 0)

        logtotal = math.log(self.word_fd.N())
        for idx in range(N - 1, -1, -1):  # 逆序规划 选择一条整个路径频率最大的句子
            route[idx] = max(
                (math.log(self.word_fd.get(sentence[idx:x + 1]) or 1) -
                 logtotal + route[x + 1][0],
                 x) for x in DAG[idx])  # x 终点索引点 idx 考察开始点

    def __cut_DAG(self, sentence):
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
                        if not self.word_fd.get(buf):  # 词典里找不到的词 用HMM来分
                            recognized = self.hmm_segment.cut(buf)
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
            elif not self.word_fd.get(buf):
                recognized = self.hmm_segment.cut(buf)
                for t in recognized:
                    yield t
            else:
                for elem in buf:
                    yield elem

    def tokenize(self, s):
        return self.lcut(s)

    def cut(self, sentence):
        """
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

    def load_userdict(self, filename):
        self.check_initialized()

        with open(filename, 'rt', encoding='utf8') as f:
            for line in f:
                word, freq, tag = re_userdict.match(line).groups()
                word = word.strip()
                if freq is not None:
                    freq = freq.strip()
                else:
                    freq = 1

                self.add_word(word, freq)

    def add_word(self, word, freq=1):
        """
        Add a word to dictionary.
        freq and tag can be omitted, freq defaults to be a calculated value
        that ensures the word can be cut out.
        """
        self.check_initialized()
        word = strdecode(word)
        freq = int(freq)

        self.word_fd.update({word: freq})

    def save_model(self, save_hmm=False):
        wlock = CACHE_WRITING.get(self.dictionary, threading.RLock())
        CACHE_WRITING[self.dictionary] = wlock
        cache_file = self._get_cache_file()
        with wlock:
            logger.debug(
                "Dumping model to file cache {0}".format(cache_file))

            set_json_value(cache_file, 'word_fd', dict(self.word_fd))
            set_json_value(cache_file, 'word_fd_timestamp', int(time.time()))

            if save_hmm:
                self.hmm_segment.save_model()

        try:
            del CACHE_WRITING[self.dictionary]
        except KeyError:
            pass
