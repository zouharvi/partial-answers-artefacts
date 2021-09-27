#!/usr/bin/env python3

import argparse
from collections import Counter

parser = argparse.ArgumentParser()    
parser.add_argument('filename', nargs='+')
parser.add_argument('--source')
args = parser.parse_args()
data = []
for filename in args.filename:
    with open(filename, "r") as f:
        data.append([x.strip() for x in f.readlines()])

with open(args.source, "r") as f:
    data_source = [x.strip() for x in f.readlines()]

data = zip(*data)

for line_i, (line, line_source) in enumerate(zip(data, data_source)):
    freq = Counter(line)
    print(freq.most_common(1)[0][0])