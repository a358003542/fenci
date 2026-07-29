# generate_result.py
import sys

from fenci import Segment

s = Segment()

def segment(text):
    res = s.lcut(text)
    return ' '.join(res)

input_path = "msr_test.utf8"
output_path = "my_result.txt"

with open(input_path, encoding='utf-8') as fin, \
        open(output_path, 'w', encoding='utf-8') as fout:
    for line in fin:
        line = line.strip()
        if not line:
            fout.write('\n')
            continue
        result = segment(line)
        fout.write(result + '\n')

print(f"结果已写入 {output_path}")


