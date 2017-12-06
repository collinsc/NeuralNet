#! /usr/bin/env python3

import sys
import re

batch_r = re.compile("batch")
int_r = re.compile("[0-9]+")
float_r = re.compile("[0-9]+\.[0-9]+")
start = False
count = 0
print("batch\tconjugate%\tvanilla%")
for line in sys.stdin.readlines():
    line = line.strip()
    if not start:
        batch = batch_r.match(line)
        if batch:
            sys.stdout.write(int_r.search(line).group())
            start = True 
    else:
        stats = float_r.findall(line)
        if (stats):
            [sys.stdout.write("\t" + stat) for stat in stats]
            if count == 3:
                sys.stdout.write("\n")
                start = False
                count = 0
            else:
                count = count + 1



