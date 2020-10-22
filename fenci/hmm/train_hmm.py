#!/usr/bin/env python
# -*-coding:utf-8-*-


import os
from nltk import FreqDist, bigrams
import nltk
from math import log

train_data_root = '../../icwb2-data/training'

msr_trainning_file = os.path.join(train_data_root, 'msr_training.utf8')
pku_trainning_file = os.path.join(train_data_root, 'pku_training.utf8')


def read_training_content():
    return open(msr_trainning_file, encoding='utf8').read() + \
           open(pku_trainning_file, encoding='utf8').read()


def output_training_dict():
    content = read_training_content()

    content_list = content.split()

    fd = FreqDist(content_list)

    with open('../training_dict.txt', 'wt') as f:
        for word, count in fd.items():
            print(f'{word} {count}', file=f)


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


def prepare_bmes_content():
    content = read_training_content()
    content_list = content.split()

    bmes_content_list = [suggest_bmes(w) for w in content_list]

    new_bmes_content_list = []

    for t in bmes_content_list:
        for s in t.split():
            new_bmes_content_list.append(nltk.tag.str2tuple(s))

    return new_bmes_content_list


P_START = [0.7689828525554734, 0.0, 0.0, 0.2310171474445266]
P_transMatrix = {'B': {'B': 0, 'E': 0, 'M': 0, 'S': 0},
                 'E': {'B': 0, 'E': 0, 'M': 0, 'S': 0},
                 'M': {'B': 0, 'E': 0, 'M': 0, 'S': 0},
                 'S': {'B': 0, 'E': 0, 'M': 0, 'S': 0}}

P_emit = {'B': {}, 'E': {}, 'M': {}, 'S': {}}


# log(0) -infinity
def train_trans_matrix():
    """
    BB BM BE BS
    MB MM ME MS
    EB EM EE ES
    SB SM SE SS
    pBM = C(BM)/C(B)
    :return:
    """
    bmes_content_list = prepare_bmes_content()

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

    for k, v in P_transMatrix.items():
        count = sum(v.values())
        for k2 in v:
            P_transMatrix[k][k2] = P_transMatrix[k][k2] / count

    new_P_transMatrix = {}

    for k in P_transMatrix:
        for k2 in P_transMatrix[k]:
            if P_transMatrix[k][k2] == 0:
                pass
            else:
                if k not in new_P_transMatrix:
                    new_P_transMatrix[k] = {}
                new_P_transMatrix[k][k2] = log(P_transMatrix[k][k2])

    with open('prob_trans.py', 'wt') as f:
        print(f"""P={new_P_transMatrix}""", file=f)


def train_emit_matrix():
    bmes_content_list = prepare_bmes_content()

    bmes_emit_list = [item[0][-1] + item[1][0] for item in
                      bigrams(bmes_content_list)]

    result_emit_fd = FreqDist(bmes_emit_list)

    for k, v in result_emit_fd.items():
        P_emit[k[0]][k[-1]] = v

    for k, v in P_emit.items():
        count = sum(v.values())
        for k2 in v:
            P_emit[k][k2] = log(P_emit[k][k2] / count)

    with open('prob_emit.py', 'wt') as f:
        print(f"""P={P_emit}""", file=f)


if __name__ == '__main__':
    output_training_dict()
    train_trans_matrix()
    train_emit_matrix()
