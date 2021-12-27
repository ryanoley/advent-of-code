# ==========================================================
#                     Advent of Code
# ==========================================================
'''
Problem Descriptions here:
https://adventofcode.com/2021

Python solutions for (as many as I can get through) days of advent of code
    2021. Attempt to use as much base python and numpy before other packages
    (e.g. pandas).  Some concepts leveraged: Recusion, networkx Graph,
    Dijkstra's shortest-path, numpy convolve, ...

'''

# Gerneric imports
import os
import numpy as np
import pandas as pd


# ==========================================================
#                     Day 1
# ==========================================================
inp_path = '~/GitHub/advent-of-code/data/1_1_input.txt'

with open(os.path.expanduser(inp_path), 'r') as file:
    inp_data = file.readlines()

inp_data = [int(x.replace('\n', '')) for x in inp_data]

((inp_data - np.roll(inp_data, 1)) > 0).sum()

df = pd.DataFrame(columns=['depth'], data=inp_data)
df['prior_depth'] = df.depth.shift(1)
((df.depth - df.prior_depth) > 0).sum()


df['depth_roll3_sum'] = df.depth.rolling(window=3).sum().shift(-2)
df['prior_depth_roll3_sum'] = df.depth_roll3_sum.shift(1)
df['delta_roll3'] = df.depth_roll3_sum - df.prior_depth_roll3_sum
(df.delta_roll3 > 0).sum()


# ==========================================================
#                     Day 2
# ==========================================================
inp_path = '~/GitHub/advent-of-code/data/2_1_input.txt'

with open(os.path.expanduser(inp_path), 'r') as file:
    inp_data = file.readlines()
inp_data = [str(x.replace('\n', '')) for x in inp_data]
inp_data = [(x.split(' ')[0], int(x.split(' ')[1])) for x in inp_data]

df = pd.DataFrame(columns=['direction', 'val'], data=inp_data)
agg = df.groupby('direction').val.sum()
(agg.down - agg.up) * agg.forward

df['aim'] = np.cumsum(np.where(df.direction == 'down', df.val, np.where(df.direction == 'up', df.val*-1, 0)))
f_df = df[df.direction == 'forward']
f_df.val.sum() * (f_df.val*f_df.aim).sum()


# ==========================================================
#                     Day 3
# ==========================================================
inp_path = '~/GitHub/advent-of-code/data/3_1_input.txt'

with open(os.path.expanduser(inp_path), 'r') as file:
    inp_data = file.readlines()
inp_data = [str(x.replace('\n', '')) for x in inp_data]

sums = [np.sum([int(x[i]) for x in inp_data]) for i in range(12)]
gamma = int(''.join(['1' if x > 500 else '0' for x in sums]), 2)
epsilon = int(''.join(['0' if x > 500 else '1' for x in sums]), 2)


sums = [np.sum([int(x[i]) for x in inp_data]) for i in range(12)]
most_common = ['1' if x > 500 else '0' for x in sums]
o2 = inp_data.copy()
i = 0
while len(o2) > 1:
    o2 = [x for x in o2 if x[i] == most_common[i]]
    i += 1
    sums = [np.sum([int(x[i]) for x in o2]) for i in range(12)]
    most_common = ['1' if x >= float(len(o2)/2) else '0' for x in sums]


sums = [np.sum([int(x[i]) for x in inp_data]) for i in range(12)]
least_common = ['0' if x >= 500 else '1' for x in sums]
co2 = inp_data.copy()
i = 0
while len(co2) > 1:
    co2 = [x for x in co2 if x[i] == least_common[i]]
    i += 1
    sums = [np.sum([int(x[i]) for x in co2]) for i in range(12)]
    least_common = ['0' if x >= float(len(co2)/2) else '1' for x in sums]

int(o2[0], 2) * int(co2[0], 2)


# ==========================================================
#                     Day 4
# ==========================================================
inp_path = '~/GitHub/advent-of-code/data/4_1_input.txt'

with open(os.path.expanduser(inp_path), 'r') as file:
    inp_data = file.readlines()
inp_data = [str(x.replace('\n', '')) for x in inp_data]

numbers = [int(x) for x in inp_data[0].split(',')]

inp_data = inp_data[2:]
board_start_ix = list(range(0, len(inp_data), 6))
board_end_ix = list(range(5, len(inp_data), 6))
board_end_ix.append(len(inp_data))


def to_np_arr(board_list):
    board_list = [x.split(' ') for x in board_list]
    return np.array([process_line(x) for x in board_list])


def process_line(line_list):
    return [int(x) for x in line_list if x != '']


def check_col_bingo(master, return_ix=False):
    col_result = np.sum(master, axis=1)
    bingo_res = np.where(col_result == -5)[0]
    if len(bingo_res)  >  0:
        row_ix = bingo_res[0]
        board_ix = row_ix -  row_ix % 5
        return board_ix if return_ix else master[board_ix:board_ix+5,:]
    return False


def check_row_bingo(master, return_ix=False):
    row_sums = np.stack([np.convolve(master[:, i], np.ones(5, int), 'valid')[::5] for i in range(5)], axis=1)
    bingo_res = np.where(row_sums == -5)[0]
    if len(bingo_res) >  0:
        row_ix = bingo_res[0] * 5
        return row_ix if return_ix else master[row_ix:row_ix+5, :]
    return False


boards = [to_np_arr(inp_data[i:j]) for i, j in zip(board_start_ix, board_end_ix)]

master = np.vstack(boards)
for n in numbers:
    master[master == n] = -1
    cb = check_col_bingo(master)
    rb = check_row_bingo(master)
    if isinstance(rb, np.ndarray) or isinstance(cb, np.ndarray):
        break
rb[rb!=-1].sum() * n


end = False
master = np.vstack(boards)
for n in numbers:
    master[master == n] = -1
    while isinstance(check_col_bingo(master, True), np.int64):
        if len(master) == 5:
            end = True
            break
        row_ix = check_col_bingo(master, True)
        master = np.delete(master, range(row_ix, row_ix + 5), 0)

    while isinstance(check_row_bingo(master, True), np.int64):
        if len(master) == 5:
            end = True
            break
        row_ix = check_row_bingo(master, True)
        master = np.delete(master, range(row_ix, row_ix + 5), 0)
    if end:
        break
master[master != 1].sum() * n

# ==========================================================
#                     Day 5
# ==========================================================
inp_path = '~/GitHub/advent-of-code/data/5_1_input.txt'

with open(os.path.expanduser(inp_path), 'r') as file:
    inp_data = file.readlines()
inp_data = [str(x.replace('\n', '')) for x in inp_data]
line_starts = [x.split(' -> ')[0].split(',') for x in inp_data]
line_ends = [x.split(' -> ')[1].split(',') for x in inp_data]

line_starts = [(int(x[0]), int(x[1])) for x in line_starts]
line_ends = [(int(x[0]), int(x[1])) for x in line_ends]

# pt1
p1_starts = []
p1_ends = []
for ls, le in zip(line_starts, line_ends):
    if ls[0] == le[0] or ls[1] == le[1]:
        p1_starts.append(ls)
        p1_ends.append(le)

max_pt = np.vstack(p1_starts + p1_ends).max(axis=0)
floor = np.zeros((max_pt[1]+1, max_pt[0]+1))

for ps, pe in zip(p1_starts, p1_ends):
    if ps[0] == pe[0]:
        start = ps if ps[1]<pe[1] else pe
        end = ps if start == pe else pe
        end = (end[0]+1, end[1]+1)
    else:
        start = ps if ps[0] < pe[0] else pe
        end = ps if start == pe else pe
        end = (end[0]+1, end[1]+1)

    floor[start[1]:end[1], start[0]:end[0]] += 1

floor[floor > 1].shape

# pt 2
max_pt = np.vstack(line_starts + line_ends).max(axis=0)
floor = np.zeros((max_pt[1]+1, max_pt[0]+1))

for ps, pe in zip(line_starts, line_ends):
    if ps[0] == pe[0]:
        start = ps if ps[1] < pe[1] else pe
        end = ps if start == pe else pe
        end = (end[0]+1, end[1]+1)
        floor[start[1]:end[1], start[0]:end[0]] += 1
    elif ps[1] == pe[1]:
        start = ps if ps[0] < pe[0] else pe
        end = ps if start == pe else pe
        end = (end[0]+1, end[1]+1)
        floor[start[1]:end[1], start[0]:end[0]] += 1
    else:
        x1 = ps[0]
        y1 = ps[1]
        x2 = pe[0]
        y2 = pe[1]
        if x1 <= x2 and y1 <= y2:
            dx_ix = list(range(x1, x2+1))
            dy_ix = list(range(y1, y2+1))
        elif x1 <= x2 and y1 > y2:
            dx_ix = list(range(x1, x2+1))
            dy_ix = list(range(y2, y1+1))[::-1]
        elif x1 > x2 and y1 < y2:
            dx_ix = list(range(x2, x1+1))[::-1]
            dy_ix = list(range(y1, y2+1))
        else:
            dx_ix = list(range(x2, x1+1))[::-1]
            dy_ix = list(range(y2, y1+1))[::-1]
        floor[(np.array(dy_ix), np.array(dx_ix))] += 1
floor[floor > 1].shape


# ==========================================================
#                     Day 6
# ==========================================================
inp_path = '~/GitHub/advent-of-code/data/6_1_input.txt'

with open(os.path.expanduser(inp_path), 'r') as file:
    inp_data = file.readlines()
inp_data = np.array([int(x) for x in inp_data[0].split(',')])

n_days = 80
fish = inp_data.copy()
for d in range(n_days):
    new_fish = (fish == 0).sum()
    fish = np.where(fish == 0, 7, fish)
    fish = np.concatenate((fish, np.repeat(9, new_fish)))
    fish -= 1
len(fish)

# pt 2
n_days = 256
fish = inp_data.copy()
fish_count = np.array([range(9), np.zeros(9)])
initial_count = np.unique(fish, return_counts=True)
fish_count[((np.repeat(1, len(initial_count[0]))), initial_count[0])] = initial_count[1]

for d in range(n_days):
    new_fish = fish_count[1, 0]
    fish_count[1, range(8)] = fish_count[1, range(1, 9)]
    fish_count[1, 8] = new_fish
    fish_count[1, 6] += new_fish

fish_count.sum(axis=1)[1]


# ==========================================================
#                     Day 7
# ==========================================================

inp_path = '~/GitHub/advent-of-code/data/7_1_test_input.txt'

with open(os.path.expanduser(inp_path), 'r') as file:
    inp_data = file.readlines()
inp_data = np.array([int(x) for x in inp_data[0].split(',')])

positions = np.array(range(min(inp_data), max(inp_data)+1))

# p1
cost = np.array([np.abs(inp_data-x).sum() for x in positions])
min_cost = cost[np.argmin(cost)]
best_position = np.argmin(cost)


# p2
cost_base = np.array([np.abs(inp_data-x) for x in positions])
cost_crab = [sum(np.arange(1, x+1).sum() for x in bc) for bc in cost_base]
min_cost = cost_crab[np.argmin(cost_crab)]
best_position = np.argmin(cost_crab)


# ==========================================================
#                     Day 8
# ==========================================================

inp_path = '~/GitHub/advent-of-code/data/8_1_input.txt'
with open(os.path.expanduser(inp_path), 'r') as file:
    inp_data = file.readlines()

inp_data0 = np.array([x.replace('\n', '').split('|')[0].split(' ')[:-1] for x in inp_data])
inp_data1 = np.array([x.replace('\n', '').split('|')[1].split(' ')[1:] for x in inp_data])

digit_nsegments = np.array([[0, 6],
                            [1, 2],
                            [2, 5],
                            [3, 5],
                            [4, 4],
                            [5, 5],
                            [6, 6],
                            [7, 3],
                            [8, 7],
                            [9, 6]])

digit_segment_layout = np.array([[0, [0, 1, 2, 4, 5, 6]],
                                 [1, [2, 5]],
                                 [2, [0, 2, 3, 4, 6]],
                                 [3, [0, 2, 3, 5, 6]],
                                 [4, [1, 2, 3, 5]],
                                 [5, [0, 1, 3, 5, 6]],
                                 [6, [0, 1, 3,  4, 5, 6]],
                                 [7, [0, 2, 5]],
                                 [8, [0, 1, 2, 3, 4, 5, 6]],
                                 [9, [0, 1, 2, 3, 5, 6]]])


# p1
inp_length = [len(x) for x in inp_data1.flatten()]
sum([x in digit_nsegments[:, 1] for x in inp_length])


# p2
def get_position_mask(seg10):
    output_mask = np.empty((7, 1), dtype=str)
    seg10_length = np.array([len(x) for x in seg10])

    digit_codes = np.array([seg10[np.where(seg10_length == x)] for x in digit_nsegments[:, 1]], dtype=object)

    pos0 = set(list(digit_codes[7][0])) - set(list(digit_codes[1][0]))
    output_mask[0] = pos0.pop()

    dig7 = np.unique(list(''.join(digit_codes[9])), return_counts=True)
    output_mask[2] = np.intersect1d(dig7[0][dig7[1] == 2], list(digit_codes[1][0]))

    pos5 = set(list(digit_codes[1][0])) - set(output_mask[2])
    output_mask[5] = pos5.pop()

    pos34 = set(dig7[0][dig7[1] == 2]) - set(output_mask[2])
    output_mask[3] = np.intersect1d(list(pos34), list(digit_codes[4][0]))[0]
    output_mask[4] = set(pos34 - set(output_mask[3])).pop()

    pos1 = set(list(digit_codes[4][0])) - set(output_mask[[2, 3, 5], 0])
    output_mask[1] = pos1.pop()

    pos6 = set(list(digit_codes[8][0])) - set(output_mask[[0, 1, 2, 3, 4, 5], 0])
    output_mask[6] = pos6.pop()
    return output_mask


def translate_to_digit(inp_n, pos_mask):
    segments = [np.where(pos_mask == x)[0][0] for x in list(inp_n)]
    segments.sort()
    for item in digit_segment_layout:
        if segments == item[1]:
            return item[0]
    return False


digit_output = []
for decode, solve in zip(inp_data0, inp_data1):
    pos_mask = get_position_mask(decode)
    output = [translate_to_digit(x, pos_mask) for x in solve]
    output = int(''.join([str(x) for x in output]))
    digit_output.append(output)


# ==========================================================
#                     Day 9
# ==========================================================

inp_path = '~/GitHub/advent-of-code/data/9_1_input.txt'
with open(os.path.expanduser(inp_path), 'r') as file:
    inp_data = file.readlines()

inp_data = [str(x.replace('\n', '')) for x in inp_data]
inp_data = np.vstack([[int(x) for x in list(y)] for y in inp_data])

# pt 1
l_arr = np.roll(inp_data, 1, axis=1)
l_arr[:, 0] = 10

r_arr = np.roll(inp_data, -1, axis=1)
r_arr[:, -1] = 10

u_arr = np.roll(inp_data, 1, axis=0)
u_arr[0, :] = 10

d_arr = np.roll(inp_data, -1, axis=0)
d_arr[-1, :] = 10

is_low_pt = (inp_data < l_arr) & (inp_data < r_arr) & (inp_data < u_arr) & (inp_data < d_arr)
(inp_data[is_low_pt] + 1).sum()


# pt2
def count_basin_range(rix, cix, visited):
    max_rix = inp_data.shape[0] - 1
    max_cix = inp_data.shape[1] - 1
    br = 0
    # left
    if cix > 0 and (rix, cix-1) not in visited:
        if (inp_data[rix, cix-1] > inp_data[rix, cix]) & (inp_data[rix, cix-1] < 9):
            br += 1
            visited.append((rix, cix-1))
            br += count_basin_range(rix, cix-1, visited)
    # right
    if cix < max_cix and (rix, cix+1) not in visited:
        if (inp_data[rix, cix+1] > inp_data[rix, cix]) & (inp_data[rix, cix+1] < 9):
            br += 1
            visited.append((rix, cix+1))
            br += count_basin_range(rix, cix+1, visited)
    # up
    if rix > 0 and (rix-1, cix) not in visited:
        if (inp_data[rix-1, cix] > inp_data[rix, cix]) & (inp_data[rix-1, cix] < 9):
            br += 1
            visited.append((rix-1, cix))
            br += count_basin_range(rix-1, cix, visited)
    # down
    if rix < max_rix and (rix+1, cix) not in visited:
        if (inp_data[rix+1, cix] > inp_data[rix, cix]) & (inp_data[rix+1, cix] < 9):
            br += 1
            visited.append((rix+1, cix))
            br += count_basin_range(rix+1, cix, visited)
    return br


low_pts = np.where(is_low_pt is True)
out = np.array([count_basin_range(r, c, []) for (r, c) in zip(low_pts[0], low_pts[1])])
out += 1
out.sort()
np.product(out[-3:])


# ==========================================================
#                     Day 10
# ==========================================================

inp_path = '~/GitHub/advent-of-code/data/10_1_input.txt'
with open(os.path.expanduser(inp_path), 'r') as file:
    inp_data = file.readlines()

inp_data = [str(x.replace('\n', '')) for x in inp_data]

openers = ['(', '[', '{', '<']
closers = [')', ']', '}', '>']
score = {')': 3, ']': 57, '}': 1197, '>': 25137}


def check_line(line):
    if line[0] in closers:
        return line[0]
    line_c_open = [line[0]]

    for i in range(1, len(line)):
        char = line[i]
        if char in openers:
            line_c_open.append(char)
            continue
        expected_open = openers[closers.index(char)]
        if line_c_open[-1] == expected_open:
            line_c_open.pop()
        else:
            return score[char]
    return 0


sum([check_line(x) for x in inp_data])

# p2
inc_mask = np.array([check_line(x) for x in inp_data]) == 0
incomplete = np.array(inp_data)[inc_mask]

score_p2 = {')': 1, ']': 2, '}': 3, '>': 4}


def complete_line_score(line):
    line_c_open = [line[0]]
    for i in range(1, len(line)):
        char = line[i]
        if char in openers:
            line_c_open.append(char)
            continue
        expected_open = openers[closers.index(char)]
        if line_c_open[-1] == expected_open:
            line_c_open.pop()
    line_c_close = [closers[openers.index(x)] for x in line_c_open[::-1]]
    char_scores = [score_p2[x] for x in line_c_close]
    out_score = 0
    for x in char_scores:
        out_score *= 5
        out_score += x
    return out_score


np.median([complete_line_score(x) for x in incomplete])


# ==========================================================
#                     Day 11
# ==========================================================

inp_path = '~/GitHub/advent-of-code/data/11_1_input.txt'
with open(os.path.expanduser(inp_path), 'r') as file:
    inp_data = file.readlines()

inp_data = [str(x.replace('\n', '')) for x in inp_data]
inp_data = np.vstack([[int(x)  for  x in list(y)] for y in inp_data])


def increment_neighbors(rix, cix, inp_data):
    max_rix = inp_data.shape[0]
    max_cix = inp_data.shape[1]

    n_ix = [(rix-1, cix), (rix-1, cix-1), (rix-1, cix+1),
            (rix, cix-1), (rix, cix+1),
            (rix+1, cix), (rix+1, cix-1), (rix+1, cix+1)]
    n_ix = [x for x in n_ix if (x[0] > -1 and x[0] < max_rix and x[1] > -1 and x[1]<max_cix)]

    rw = np.array([x[0] for x in n_ix])
    cl = np.array([x[1] for x in n_ix])
    inp_data[(rw, cl)] += 1

    return inp_data


iter_arr = inp_data.copy()
flash_count = 0
for i in range(100):
    iter_arr += 1
    flash = np.where(iter_arr == 10)
    flash = set(zip(flash[0], flash[1]))
    flashed = set()

    while(len(flash)>0):
        for rix, cix in flash:
            increment_neighbors(rix, cix, iter_arr)
        flashed.update(flash)
        flash = np.where(iter_arr >= 10)
        flash = set(zip(flash[0], flash[1]))
        flash = flash - flashed
    iter_arr[iter_arr >= 10] = 0
    flash_count += len(flashed)
    # pt 2
    # if len(flashed) == 100:
    #     break


# ==========================================================
#                     Day 12
# ==========================================================
import networkx as nx
from pyvis.network import Network


def draw_graph(G):
    net = Network(notebook=True, height="750px", width="100%")
    net.from_nx(G)
    return net.show("network.html")


inp_path = '~/GitHub/advent-of-code/data/12_1_input.txt'
with open(os.path.expanduser(inp_path), 'r') as file:
    inp_data = file.readlines()

inp_data = [tuple(str(x.replace('\n', '')).split('-')) for x in inp_data]

G = nx.Graph()
G.add_edges_from(inp_data)

draw_graph(G)


def get_next_hop(node, visited, pt2=False):
    neighbors = list(G.neighbors(node))
    if 'end' in neighbors:
        neighbors.remove('end')
        neighbors.append('end')

    if pt2 and max(visited.values()) <= 1:
        visited = {'start':1}

    return list(set(neighbors) - set(visited.keys()))


def build_paths(current_node, inp_path, inp_visited, all_paths):
    path = inp_path.copy()
    visited = inp_visited.copy()
    path.append(current_node)

    if current_node.islower():
        if current_node not in visited:
            visited[current_node] = 1
        else:
            visited[current_node] += 1

    if current_node == 'end':
        all_paths.append(path)
        return all_paths

    next_hop = get_next_hop(current_node, visited, True)

    for nh in next_hop:
        all_paths = build_paths(nh, path, visited, all_paths)

    return all_paths


paths = build_paths('start', [], {}, [])


# ==========================================================
#                     Day 13
# ==========================================================

inp_path = '~/GitHub/advent-of-code/data/13_1_input.txt'
with open(os.path.expanduser(inp_path), 'r') as file:
    inp_data = file.readlines()

inp_data = [str(x.replace('\n', '')) for x in inp_data]
dot_ixs = np.vstack([[int(x) for x in y.split(',')] for y in inp_data[:-13]])

folds = inp_data[-12:]

paper = np.zeros((np.max(dot_ixs, axis=0)[1]+1, np.max(dot_ixs, axis=0)[0]+1))


def plot_dots(paper, dot_ixs):
    for dot in dot_ixs:
        paper[dot[1], dot[0]] = 1
    return paper


def fold(paper, fold_ix, dot_ixs, y_fold=True):
    fold_axis = 1 if y_fold else 0
    if y_fold:
        fold_axis = 1
        new_paper = np.zeros((fold_ix, paper.shape[1]))
    else:
        fold_axis = 0
        new_paper = np.zeros((paper.shape[0],fold_ix))
    ndot_ixs = dot_ixs.copy()
    for dot in ndot_ixs:
        dot_axis_val = dot[fold_axis]
        if dot_axis_val > fold_ix:
            nix = fold_ix - (dot_axis_val - fold_ix)
            dot[fold_axis] = nix
    new_paper = plot_dots(new_paper,  ndot_ixs)
    return new_paper, ndot_ixs


n_paper, ndot_ixs = fold(paper, 655, dot_ixs, False)

fold_inst = []
for f in folds:
    match = re.search(r'([xy])=(\d+)', f)
    fold_inst.append((match.groups()[0], match.groups()[1]))

for f in fold_inst:
    y_fold = True if f[0] == 'y' else False
    paper, dot_ixs = fold(paper, int(f[1]), dot_ixs, y_fold)


# ==========================================================
#                     Day 14
# ==========================================================

from tqdm import tqdm

inp_path = '~/GitHub/advent-of-code/data/14_1_input.txt'
with open(os.path.expanduser(inp_path), 'r') as file:
    inp_data = file.readlines()

inp_data = [str(x.replace('\n', '')) for x in inp_data]

polymer = inp_data[0]
rules = inp_data[2:].copy()
rules = {x[:2]: ''.join([x[0], x[-1], x[1]]) for x in rules}

n_steps = 40
for s in tqdm(range(n_steps)):
    polymer_pairs = [polymer[i: i+2] for i in range(len(polymer))][:-1]
    polymer_pairs = [rules[x] if x in rules else x for x in polymer_pairs]
    polymer = ''.join([x[:-1] for x in polymer_pairs[:-1]] + polymer_pairs[-1:])

char_counts = np.unique(list(polymer), return_counts=True)[1]
max(char_counts) - min(char_counts)


pair_dict = {}
polymer_pairs = [polymer[i: i+2] for i in range(len(polymer))][:-1]
for pp in polymer_pairs:
    if pp in pair_dict:
        pair_dict[pp] += 1
    else:
        pair_dict[pp] = 1

n_steps = 40
for s in range(n_steps):
    char_pairs = list(pair_dict.keys()).copy()
    char_vals = list(pair_dict.values()).copy()
    pair_dict = {}
    for (cp, cv) in zip(char_pairs, char_vals):
        if cp not in rules:
            if cp in pair_dict:
                pair_dict[cp] += cv
            else:
                cp[cp] = cv
            continue

        np1 = rules[cp][:2]
        np2 = rules[cp][1:]
        if np1 in pair_dict:
            pair_dict[np1] += cv
        else:
            pair_dict[np1] = cv

        if np2 in pair_dict:
            pair_dict[np2] += cv
        else:
            pair_dict[np2] = cv

char_counts = {}
for c, v in pair_dict.items():
    if c[0] in char_counts:
        char_counts[c[0]] += v
    else:
        char_counts[c[0]] = v

char_counts[polymer[-1]] += 1
max(char_counts.values()) - min(char_counts.values())


# ==========================================================
#                     Day 15
# ==========================================================
import networkx as nx

inp_path = '~/GitHub/advent-of-code/data/15_1_input.txt'
with open(os.path.expanduser(inp_path), 'r') as file:
    inp_data = file.readlines()

inp_data = [str(x.replace('\n', '')) for x in inp_data]
inp_data = np.vstack([[int(x) for x in list(y)] for y in inp_data])

cave_arr = inp_data.copy()


#  Recursive attempt (fail)

def get_next_hops(pt, path, end_pt=(99,99)):
    hops = [(pt[0], pt[1]+1), (pt[0]-1, pt[1]), (pt[0], pt[1]-1),
            (pt[0]+1, pt[1])]

    hops = [x for x in hops if x[0]>=0 and x[0]<=end_pt[0] and x[1]>=0 and x[1]<=end_pt[0]]

    if end_pt in hops:
        hops.remove(end_pt)
        hops.append(end_pt)

    return list(set(hops) - set(path))


def build_paths(current_pt, inp_path, inp_path_score, all_scores, all_paths):
    path = inp_path.copy()
    path_score = inp_path_score
    path.append(current_pt)
    path_score += cave_arr[current_pt]

    max_ix = cave_arr.shape[0] - 1

    if path_score > min(all_scores):
        return all_paths, all_scores
    elif current_pt == (max_ix, max_ix):
        all_paths.append(path)
        all_scores.append(path_score)
        return all_paths, all_scores

    next_hop = get_next_hops(current_pt, path, (max_ix, max_ix))

    for nh in next_hop:
        all_paths, all_scores = build_paths(nh, path, path_score, all_scores, all_paths)

    return all_paths, all_scores

# start_score = cave_arr.sum(axis=0)[0] + cave_arr.sum(axis=0)[-1] - cave_arr[0, -1]
# all_paths, all_scores = build_paths((0,0), [], 0, [start_score], [])


# Graph solution (Dijkstra)
def build_cave_graph(inp_arr):
    G = nx.DiGraph()
    for rix in range(inp_arr.shape[0]):
        for cix in range(inp_arr.shape[1]):
            G.add_node(f'{rix}-{cix}')

            if rix-1 >= 0:
                G.add_edge(f'{rix-1}-{cix}', f'{rix}-{cix}',
                           weight=inp_arr[rix, cix])
                G.add_edge(f'{rix}-{cix}', f'{rix-1}-{cix}',
                           weight=inp_arr[rix-1, cix])

            if cix-1 >= 0:
                G.add_edge(f'{rix}-{cix-1}', f'{rix}-{cix}',
                           weight=inp_arr[rix, cix])
                G.add_edge(f'{rix}-{cix}', f'{rix}-{cix-1}',
                           weight=inp_arr[rix, cix-1])
    return G


# p1
small_cave_graph = build_cave_graph(cave_arr)
nx.shortest_path(small_cave_graph, '0-0', '9-9', 'weight')
nx.shortest_path_length(small_cave_graph, '0-0', '9-9', 'weight')

# p2
big_cave = cave_arr.copy()
cave_tile = cave_arr.copy()
for i in range(4):
    cave_tile += 1
    cave_tile = np.where(cave_tile>9, 1, cave_tile)
    big_cave = np.concatenate((big_cave, cave_tile), axis=1)

cave_tile = big_cave.copy()
for i in range(4):
    cave_tile += 1
    cave_tile = np.where(cave_tile>9, 1, cave_tile)
    big_cave = np.concatenate((big_cave, cave_tile), axis=0)

big_cave_graph = build_cave_graph(big_cave)
nx.shortest_path(big_cave_graph, '0-0', '499-499', 'weight')
nx.shortest_path_length(big_cave_graph, '0-0', '499-499', 'weight')


# ==========================================================
#                     Day 16
# ==========================================================
inp_path = '~/GitHub/advent-of-code/data/16_1_test_input.txt'
with open(os.path.expanduser(inp_path), 'r') as file:
    inp_data = file.readlines()

inp_data = [str(x.replace('\n', '')) for x in inp_data]
inp_data = np.vstack([[int(x) for x in list(y)] for y in inp_data])

cave_arr = inp_data.copy()










