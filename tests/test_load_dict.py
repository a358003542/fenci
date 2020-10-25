#!/usr/bin/env python
# -*-coding:utf-8-*-


def test_load_dict():
    from fenci.segment import Segment
    s = Segment()

    res1 = s.tokenize(
        "机器学习是一门新型的计算机学科。")

    s.load_userdict('tests/test_dict.txt')

    res2 = s.tokenize("机器学习是一门新型的计算机学科。")

    assert '机器学习' not in res1
    assert '机器学习' in res2
