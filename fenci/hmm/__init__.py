#!/usr/bin/env python
# -*-coding:utf-8-*-

import re
import pickle
import sys

from ..utils import get_module_res, strdecode

default_encoding = sys.getfilesystemencoding()

xrange = range

iterkeys = lambda d: iter(d.keys())
itervalues = lambda d: iter(d.values())
iteritems = lambda d: iter(d.items())


def resolve_filename(f):
    try:
        return f.name
    except AttributeError:
        return repr(f)


MIN_FLOAT = -3.14e100

PrevStatus = {
    'B': 'ES',
    'M': 'MB',
    'S': 'SE',
    'E': 'BM'
}

from .prob_start import P as start_P
from .prob_trans import P as trans_P
from .prob_emit import P as emit_P


def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]  # tabular
    path = {}
    for y in states:  # init
        V[0][y] = start_p[y] + emit_p[y].get(obs[0], MIN_FLOAT)
        path[y] = [y]
    for t in xrange(1, len(obs)):
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


def __cut(sentence):
    global emit_P
    prob, pos_list = viterbi(sentence, 'BMES', start_P, trans_P, emit_P)
    begin, nexti = 0, 0
    # print pos_list, sentence
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


re_han = re.compile("([\u4E00-\u9FD5]+)")
re_skip = re.compile("([a-zA-Z0-9]+(?:\.\d+)?%?)")


def cut(sentence):
    sentence = strdecode(sentence)
    blocks = re_han.split(sentence)
    for blk in blocks:
        if re_han.match(blk):
            for word in __cut(blk):
                yield word

        else:
            tmp = re_skip.split(blk)
            for x in tmp:
                if x:
                    yield x
