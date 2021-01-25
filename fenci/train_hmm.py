#!/usr/bin/env python
# -*-coding:utf-8-*-

import os
from simple_nltk import FreqDist
from simple_nltk.util import bigrams
from simple_nltk.tag.util import str2tuple
from math import log

from fenci.utils import read_training_content


def suggest_bmes(word):
    if len(word) == 1:
        return f'{word}/S'
    elif len(word) == 2:
        return f'{word[0]}/B {word[1]}/E'
    elif len(word) == 3:
        return f'{word[0]}/B {word[1]}/M {word[2]}/E'
    elif len(word) > 3:
        result = f'{word[0]}/B '
        for s in word[1:-1]:
            result += f'{s}/M '
        result += f'{word[-1]}/E'
        return result
    else:
        print(f'wrong word length !!!!')


def prepare_bmes_content(root, regexp):
    content = read_training_content(root, regexp)
    content_list = content.split()

    bmes_content_list = [suggest_bmes(w) for w in content_list]

    new_bmes_content_list = []

    for t in bmes_content_list:
        for s in t.split():
            new_bmes_content_list.append(str2tuple(s))

    return new_bmes_content_list


def train_trans_matrix(root, regexp):
    """
    BB BM BE BS
    MB MM ME MS
    EB EM EE ES
    SB SM SE SS
    pBM = C(BM)/C(B)
    :return:
    """
    P_transMatrix = {'B': {'B': 0, 'E': 0, 'M': 0, 'S': 0},
                     'E': {'B': 0, 'E': 0, 'M': 0, 'S': 0},
                     'M': {'B': 0, 'E': 0, 'M': 0, 'S': 0},
                     'S': {'B': 0, 'E': 0, 'M': 0, 'S': 0}}
    bmes_content_list = prepare_bmes_content(root, regexp)

    bmes_list = [item[0][-1] + item[1][-1] for item in
                 bigrams(bmes_content_list)]

    result_fd = FreqDist(bmes_list)

    from itertools import product
    for key in map(lambda a: a[0] + a[1],
                   product(['B', 'M', 'E', 'S'], repeat=2)):
        if key in result_fd:
            a = key[0]
            b = key[1]
            P_transMatrix[a][b] = result_fd[key]

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


def train_trans_matrix_to_file(root, regexp, output_dir='.'):
    P_transMatrix = train_trans_matrix(root, regexp)

    for k, v in P_transMatrix.items():
        count = sum(v.values())
        for k2 in v:
            P_transMatrix[k][k2] = log(P_transMatrix[k][k2] / count)

    with open(os.path.join(output_dir, 'hmm/prob_trans.py'), 'wt',
              encoding='utf8') as f:
        print(f"""P={P_transMatrix}""", file=f)


def train_emit_matrix(root, regexp):
    P_emit = {'B': {}, 'E': {}, 'M': {}, 'S': {}}
    bmes_content_list = prepare_bmes_content(root, regexp)

    bmes_emit_list = [item[0][-1] + item[1][0] for item in
                      bigrams(bmes_content_list)]

    result_emit_fd = FreqDist(bmes_emit_list)

    for k, v in result_emit_fd.items():
        P_emit[k[0]][k[-1]] = v

    return P_emit


def train_emit_matrix_to_file(root, regexp, output_dir='.'):
    P_emit = train_emit_matrix(root, regexp)
    for k, v in P_emit.items():
        count = sum(v.values())
        for k2 in v:
            P_emit[k][k2] = log(P_emit[k][k2] / count)

    with open(os.path.join(output_dir, 'hmm/prob_emit.py'), 'wt',
              encoding='utf8') as f:
        print(f"""P={P_emit}""", file=f)


if __name__ == '__main__':
    root = 'icwb2-data/training'
    regexp = '(?!\.).*\.utf8'

    train_trans_matrix_to_file(root, regexp)
    train_emit_matrix_to_file(root, regexp)
