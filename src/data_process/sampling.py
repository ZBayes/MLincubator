# 抽样计算
import sys
sys.path.append("../util")
from util import logTool
import random

INPUTPATH = "../../data/baseProcess"
OUTPUTPATH = "../../data/sample_30"
LOG = logTool("../../data/log/sampling")

sample_rate = 0.3

fout = open(OUTPUTPATH, "w")
with open(INPUTPATH, "r") as f:

    index = 0
    numbers = 0
    for line in f:
        if random.random() <= sample_rate:
            numbers = numbers + 1
            fout.write(line)
        index = index + 1
        if index % 1000 == 0:
            print("we have scan %s data and %s of them is seleted" %
                  (index, numbers))
            LOG.info("we have scan %s data and %s of them is seleted" %
                     (index, numbers))
fout.close()
