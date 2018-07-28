# 对原始数据进行预处理，以便进行后续计算

import sys
sys.path.append("../util")
from util import csvHelper, logTool

INPUTPATH = "../../data/train_set.csv"
OUTPUTPATH = "../../data/baseProcess"
LOG = logTool("../../data/log/baseProcess")

CSVHELPER_INPUT = csvHelper(INPUTPATH)
f = open(OUTPUTPATH, mode="w")

dataCount = 0
for dataItem in CSVHELPER_INPUT.csvRead():
    if dataCount == 0:
        dataCount = dataCount + 1
        LOG.info("start calculating")
        continue
    try:
        sentence = dataItem[2]  # 句子
        label = dataItem[3]     # 标签
        f.writelines("%s,%s\n" % (label, sentence))
    except Exception as e:
        print("Error: %s,\t,%s" % (Exception, e))
        LOG.error("Error: %s,\t,%s" % (Exception, e))
    finally:
        if dataCount % 5000 == 0:
            print("now we have counted %s dataItems" % dataCount)
            LOG.info("now we have counted %s dataItems" % dataCount)
    dataCount = dataCount + 1

print("completed")


f.close()
LOG.close()
