# word2vector测试
import sys
sys.path.append("../util")
from util import logTool
from gensim.models import Word2Vec
import numpy as np


def calSenVec(model, sentense):
    wordMat = []
    for word in sentense:
        if word in model.wv.vocab:
            wordMat.append(model[word])
    wordMat = np.array(wordMat)
    senVec = np.mean(wordMat, axis=0)
    return senVec


if __name__ == '__main__':
    INPUTPATH = "../../data/sample_10"
    MODELSAVEDPATH = "../../data/model/Word2Vec2018722_1.model"
    OUPUTPATH = "../../data/sample_10_word2Vec_AVG_100"

    model = Word2Vec.load(MODELSAVEDPATH)
    print(model)
    # print(len(model.wv.vocab))
    # print(model['1138901'])
    fout = open(OUPUTPATH, "w")
    with open(INPUTPATH, "r") as f:
        index = 0
        for line in f:
            indexLength = 0
            ll = line.strip()
            label, sentense = ll.split(",")
            sentense = sentense.split(" ")
            senVec = calSenVec(model, sentense)
            fout.write(label + "," + " ".join(str(i) for i in senVec.tolist())+"\n")
            index = index + 1
            if index % 1000 == 0:
                print(index)
    fout.close()

    INPUTPATH = "../../data/sample_10_test"
    MODELSAVEDPATH = "../../data/model/Word2Vec2018722_1.model"
    OUPUTPATH = "../../data/sample_10_word2Vec_AVG_100_test"
    LOG = logTool("../../data/log/word2vector")

    model = Word2Vec.load(MODELSAVEDPATH)
    print(model)
    # print(len(model.wv.vocab))
    # print(model['1138901'])
    fout = open(OUPUTPATH, "w")
    with open(INPUTPATH, "r") as f:
        index = 0
        for line in f:
            indexLength = 0
            ll = line.strip()
            label, sentense = ll.split(",")
            sentense = sentense.split(" ")
            senVec = calSenVec(model, sentense)
            fout.write(label + "," + " ".join(str(i) for i in senVec.tolist())+"\n")
            index = index + 1
            if index % 1000 == 0:
                print(index)
    fout.close()
