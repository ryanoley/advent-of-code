# ==========================================================
#                     Advent of Code
# ==========================================================
'''
Problem Descriptions here:
https://adventofcode.com/2022

Python solutions for (as many as I can get through) days of advent of code
    2022. Attempt to use as much base python and numpy before other packages
    (e.g. pandas).

'''

# Gerneric imports
import os
import numpy as np
import pandas as pd


# ==========================================================
#                     Day 1
# ==========================================================
inp_path = '~/repos/advent-of-code/2022/data/1_input.txt'

with open(os.path.expanduser(inp_path), 'r') as file:
    inp_data = file.readlines()
inp_data = [x.replace('\n', '') for x in inp_data]

elf_cals = []
x_cal = 0
for x in inp_data:
    if x == "":
        elf_cals.append(x_cal)
        x_cal = 0
    else:
        x_cal += int(x)



