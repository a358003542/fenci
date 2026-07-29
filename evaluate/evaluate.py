#!/usr/bin/env python3

"""中文分词评测：Precision / Recall / F1 + 速度（KB/s）"""

import sys
import time
import os


def to_region(seg_line: str):
    """将空格分词结果转为 (start, end) 区间集合"""
    words = seg_line.strip().split()
    regions = set()
    start = 0
    for w in words:
        end = start + len(w)
        regions.add((start, end))
        start = end
    return regions


def evaluate(gold_path: str, pred_path: str):
    """逐行对比金标准与预测，统计 P/R/F1"""
    total_gold = 0
    total_pred = 0
    correct = 0
    line_no = 0

    with open(gold_path, encoding='utf-8') as fg, \
            open(pred_path, encoding='utf-8') as fp:
        for g_line, p_line in zip(fg, fp):
            line_no += 1
            g_reg = to_region(g_line)
            p_reg = to_region(p_line)
            total_gold += len(g_reg)
            total_pred += len(p_reg)
            correct += len(g_reg & p_reg)

    # 行数校验
    fg_lines = sum(1 for _ in open(gold_path, encoding='utf-8'))
    fp_lines = sum(1 for _ in open(pred_path, encoding='utf-8'))
    if fg_lines != fp_lines:
        print(f"⚠️ 警告：金标准 {fg_lines} 行 vs 预测 {fp_lines} 行，行数不一致！")

    precision = correct / total_pred * 100 if total_pred else 0
    recall = correct / total_gold * 100 if total_gold else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print("=== 分词评测结果 ===")
    print(f"总词数(金标准): {total_gold}")
    print(f"总词数(预测):   {total_pred}")
    print(f"正确词数:       {correct}")
    print(f"Precision:      {precision:.2f}%")
    print(f"Recall:         {recall:.2f}%")
    print(f"F1:             {f1:.2f}%")


def speed_test(text_path: str, segment_fn, repeat: int = 3):
    """
    速度测试：读入文本，重复跑 repeat 次取平均
    text_path: 原始未分词文本
    segment_fn: 你的分词函数 text -> str
    """
    with open(text_path, encoding='utf-8') as f:
        texts = [l.strip() for l in f if l.strip()]

    total_bytes = sum(len(t.encode('utf-8')) for t in texts)

    times = []
    for i in range(repeat):
        start = time.time()
        for t in texts:
            segment_fn(t)
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)
    speed_kbs = total_bytes / 1024 / avg_time if avg_time > 0 else 0

    print(f"\n=== 速度测试 ===")
    print(f"文本大小:       {total_bytes / 1024:.1f} KB")
    print(f"重复次数:       {repeat}")
    print(f"平均耗时:       {avg_time:.3f} s")
    print(f"速度:           {speed_kbs:.1f} KB/s")


if __name__ == "__main__":
    gold_path = "msr_test_gold.utf8"
    pred_path = "my_result.txt"

    evaluate(gold_path, pred_path)

    from generate_result import segment

    speed_test("msr_test.utf8", segment, repeat=3)
