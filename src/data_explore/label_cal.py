# 探索标签比例
# 抽样计算
import sys
sys.path.append("../util")
from util import logTool
import random

INPUTPATH = "../../data/sample_10"

with open(INPUTPATH, "r") as f:
    labels = {}
    for line in f:
        dataItem = line.split(",")
        label = dataItem[0]
        if label in labels:
            labels[label] = labels[label] + 1
        else:
            labels[label] = 1

print(labels)
# {
#     '1': 537,
#     '9': 757,
#     '8': 699,
#     '15': 734,
#     '18': 686,
#     '3': 823,
#     '4': 377,
#     '11': 374,
#     '19': 554,
#     '14': 651,
#     '7': 297,
#     '12': 535,
#     '6': 681,
#     '13': 773,
#     '17': 276,
#     '10': 483,
#     '5': 237,
#     '16': 307,
#     '2': 306
# }
