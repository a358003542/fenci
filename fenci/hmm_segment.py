#!/usr/bin/env python
# -*-coding:utf-8-*-

import time
import re
import os
from copy import deepcopy
import logging
import threading
from math import log
from simple_nltk.tokenize.api import TokenizerI

from .base import BaseSegment
from fenci.train_hmm import train_emit_matrix, train_trans_matrix
from fenci.utils import strdecode, get_json_value, set_json_value
from fenci.const import CACHE_WRITING, DEFAULT_HMM_DATA
from fenci import __softname__

logger = logging.getLogger(__name__)

start_P = {'B': -0.26268660809250016,
           'E': -3.14e+100,
           'M': -3.14e+100,
           'S': -1.4652633398537678}

re_han_hmm = re.compile("([\u4E00-\u9FD5]+)")
re_skip_hmm = re.compile("([a-zA-Z0-9]+(?:\.\d+)?%?)")

MIN_FLOAT = -3.14e100

PrevStatus = {
    'B': 'ES',
    'M': 'MB',
    'S': 'SE',
    'E': 'BM'
}


class HMMSegment(TokenizerI, BaseSegment):
    def __init__(self, traning_root=None,
                 traning_regexp='.*\.txt', traning_mode='update',
                 cache_file=None):
        self.lock = threading.RLock()

        self.training_root = traning_root
        self.training_regexp = traning_regexp

        self.training_mode = traning_mode

        assert self.training_mode in ['update', 'replace']

        self.cache_file = cache_file
        self.tmp_dir = None

        self.P_trans = None
        self.model_data = {}
        self.P_emit = None

        self.initialized = False

    def __cut(self, sentence):
        self.check_initialized()

        prob, pos_list = viterbi(sentence, 'BMES', start_P, self.P_trans,
                                 self.P_emit)
        begin, nexti = 0, 0
        # logger.debug pos_list, sentence
        for i, char in enumerate(sentence):
            pos = pos_list[i]
            if pos == 'B':
                begin = i
            elif pos == 'E':
                yield sentence[begin:i + 1]
                nexti = i + 1
            elif pos == 'S':
                yield char
                nexti = i + 1
        if nexti < len(sentence):
            yield sentence[nexti:]

    def cut(self, sentence):
        sentence = strdecode(sentence)
        blocks = re_han_hmm.split(sentence)
        for blk in blocks:
            if re_han_hmm.match(blk):
                for word in self.__cut(blk):
                    yield word

            else:
                tmp = re_skip_hmm.split(blk)
                for x in tmp:
                    if x:
                        yield x

    def lcut(self, s):
        return list(self.cut(s))

    def tokenize(self, s):
        return self.lcut(s)

    def save_model(self):
        wlock = CACHE_WRITING.get('hmm_data', threading.RLock())
        CACHE_WRITING['hmm_data'] = wlock
        cache_file = self._get_cache_file()

        with wlock:
            logger.debug(
                "Dumping HMM model to file cache {0}".format(cache_file))
            set_json_value(cache_file, 'P_emit', self.model_data.get('P_emit'))
            set_json_value(cache_file, 'P_trans',
                           self.model_data.get('P_trans'))
            set_json_value(cache_file, 'hmm_timestamp', int(time.time()))

        try:
            del CACHE_WRITING['hmm_data']
        except KeyError:
            pass

    def training(self, root=None, regexp=None, training_mode='update'):
        assert training_mode in ['update', 'replace']

        if root is None and self.training_root is None:
            raise Exception('please give the training data root')
        root = root if root is not None else self.training_root
        regexp = regexp if regexp is not None else self.training_regexp
        training_mode = training_mode if training_mode is not None else self.training_mode

        if training_mode == 'update':
            P_emit = train_emit_matrix(root, regexp)
            P_trans = train_trans_matrix(root, regexp)
            old_P_trans = self.model_data.get('P_trans')
            old_P_emit = self.model_data.get('P_emit')

            new_P_emit = self.merge_P_emit(P_emit, old_P_emit)
            new_P_trans = self.merge_P_trans(P_trans, old_P_trans)
            self.model_data = {'P_emit': new_P_emit, 'P_trans': new_P_trans}
            self.P_emit = self._prepare_P_emit()
            self.P_trans = self._prepare_P_trans()

        elif training_mode == 'replace':
            P_emit = train_emit_matrix(root, regexp)
            P_trans = train_trans_matrix(root, regexp)
            self.model_data = {'P_emit': P_emit, 'P_trans': P_trans}
            self.P_emit = self._prepare_P_emit()
            self.P_trans = self._prepare_P_trans()

    def merge_P_trans(self, one, two):
        P_transMatrix = {'B': {'B': 0, 'E': 0, 'M': 0, 'S': 0},
                         'E': {'B': 0, 'E': 0, 'M': 0, 'S': 0},
                         'M': {'B': 0, 'E': 0, 'M': 0, 'S': 0},
                         'S': {'B': 0, 'E': 0, 'M': 0, 'S': 0}}

        from itertools import product
        for key in map(lambda a: a[0] + a[1],
                       product(['B', 'M', 'E', 'S'], repeat=2)):
            a = key[0]
            b = key[1]
            if a in one and b in one[a]:
                P_transMatrix[a][b] += one[a][b]
            if a in two and b in two[a]:
                P_transMatrix[a][b] += two[a][b]

        new_P_transMatrix = {}

        for k in P_transMatrix:
            for k2 in P_transMatrix[k]:
                if P_transMatrix[k][k2] == 0:
                    pass
                else:
                    if k not in new_P_transMatrix:
                        new_P_transMatrix[k] = {}
                    new_P_transMatrix[k][k2] = P_transMatrix[k][k2]
        return new_P_transMatrix

    def merge_P_emit(self, one, two):
        P_emit = {'B': {}, 'E': {}, 'M': {}, 'S': {}}

        for k, v in one.items():
            for word in v:
                P_emit[k][word] = v[word]

        for k, v in two.items():
            for word in v:
                if word in P_emit:
                    P_emit[k][word] += v[word]
                else:
                    P_emit[k][word] = v[word]

        return P_emit

    def initialize(self):
        with self.lock:
            if self.initialized:
                return

            t1 = time.time()

            cache_file = self._get_cache_file()

            # use cache data
            use_cache_data = False
            if os.path.isfile(cache_file):
                hmm_timestamp = get_json_value(cache_file,
                                               'hmm_timestamp')
                if hmm_timestamp:
                    use_cache_data = True

            if use_cache_data:
                logger.debug(
                    "Loading HMM model from cache {0}".format(cache_file))
                P_trans = get_json_value(cache_file, 'P_trans')
                P_emit = get_json_value(cache_file, 'P_emit')
                self.model_data = {'P_emit': P_emit, 'P_trans': P_trans}

                self.P_emit = self._prepare_P_emit()
                self.P_trans = self._prepare_P_trans()
            else:
                P_trans = get_json_value(self._get_default_model_file(),
                                         'P_trans')
                P_emit = get_json_value(self._get_default_model_file(),
                                        'P_emit')
                self.model_data = {'P_emit': P_emit, 'P_trans': P_trans}

                self.P_emit = self._prepare_P_emit()
                self.P_trans = self._prepare_P_trans()

                self.save_model()

            self.initialized = True
            logger.debug(
                "Loading model cost %.3f seconds." % (time.time() - t1))
            logger.debug("Prefix dict has been built succesfully.")

    def _get_default_model_file(self):
        from pkg_resources import resource_filename
        return resource_filename(__softname__, DEFAULT_HMM_DATA)

    def _prepare_P_trans(self):
        P_trans_data = self.model_data.get('P_trans')

        P_trans = deepcopy(P_trans_data)

        for k, v in P_trans.items():
            count = sum(v.values())
            for k2 in v:
                P_trans[k][k2] = log(P_trans[k][k2] / count)

        return P_trans

    def _prepare_P_emit(self):
        P_emit_data = self.model_data.get('P_emit')

        P_emit = deepcopy(P_emit_data)

        for k, v in P_emit.items():
            count = sum(v.values())
            for k2 in v:
                P_emit[k][k2] = log(P_emit[k][k2] / count)

        return P_emit


def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]  # tabular
    path = {}
    for y in states:  # init
        V[0][y] = start_p[y] + emit_p[y].get(obs[0], MIN_FLOAT)
        path[y] = [y]
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}
        for y in states:
            em_p = emit_p[y].get(obs[t], MIN_FLOAT)
            (prob, state) = max(
                [(V[t - 1][y0] + trans_p[y0].get(y, MIN_FLOAT) + em_p, y0) for
                 y0 in PrevStatus[y]])
            V[t][y] = prob
            newpath[y] = path[state] + [y]
        path = newpath

    (prob, state) = max((V[len(obs) - 1][y], y) for y in 'ES')

    return (prob, path[state])
