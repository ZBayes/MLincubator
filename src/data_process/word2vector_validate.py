# word2vector测试
import sys
sys.path.append("../util")
from util import logTool
from gensim.models import Word2Vec

INPUTPATH = "../../data/sample_10"
MODELSAVEDPATH = "../../data/model/Word2Vec2018722_1.model"
LOG = logTool("../../data/log/word2vector")

model = Word2Vec.load(MODELSAVEDPATH)
# print(model.vocabulary.__dict__)
print(len(model.wv.vocab))
print(model['1138901'])