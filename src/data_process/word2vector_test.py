# 抽样计算
import sys
sys.path.append("../util")
from util import logTool
from gensim.models import Word2Vec

INPUTPATH = "../../data/sample_10"
MODELSAVEDPATH = "../../data/model/Word2Vec2018722_1.model"
LOG = logTool("../../data/log/word2vector")

sentenses = []
labels = []
LOG.info("Starting loading data")
with open(INPUTPATH, "r") as f:
    for line in f:
        label,sentense = line.split(",")
        labels.append(label)
        sentenses.append(sentense.split(" "))
    print("finish loading data")
    LOG.info("finish loading data")
# print(sentenses)
LOG.info("starting training model")
model = Word2Vec(sentenses, 
                 size=1000, 
                 window=5,
                 workers=2)
model.save(MODELSAVEDPATH)
LOG.info("finish training model")


# print(model["986174"])
# print(model)