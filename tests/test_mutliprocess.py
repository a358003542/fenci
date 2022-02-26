#!/usr/bin/env python
# -*-coding:utf-8-*-

from fenci import Segment

import threading

result = []


def segment_test(i):
    segment = Segment()
    res = segment.lcut(
        '据 CNBC 报道，Google    前 CEO、Alphabet 前执行董事 Eric Schmidt 近日在参加旧金山的某高级私人活动时表示，未来十年将有两个截然不同的互联网：一个由美国领导，另一个由中国领导。。。')
    print(f'thread or process: {i} working...')

    assert res == ['据', ' ', 'CNBC', ' ', '报道', '，', 'Google', '    ', '前', ' ',
                   'CEO', '、', 'Alphabet', ' ', '前', '执行', '董事', ' ', 'Eric',
                   ' ', 'Schmidt',
                   ' ', '近日', '在', '参加', '旧金山', '的', '某', '高级', '私人', '活动', '时',
                   '表示', '，', '未来', '十年', '将', '有', '两个', '截然不同', '的',
                   '互联网', '：', '一个', '由', '美国', '领导', '，', '另', '一个', '由', '中国',
                   '领导', '。', '。', '。']


def test_multithread():
    threads = []
    for i in range(8):
        t = threading.Thread(target=segment_test, args=(i,))
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()


def test_multiprocess():
    from multiprocessing import Pool
    with Pool(5) as p:
        p.map(segment_test, range(0, 8))
