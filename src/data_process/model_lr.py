import sys
sys.path.append("../util")
from util import logTool
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    # INPUTPATH = "../../data/sample_10"
    # MODELSAVEDPATH = "../../data/model/Word2Vec2018722_1.model"
    INPUTPATH = "../../data/sample_30"
    INPUTPATH_TEST = "../../data/sample_10"
    LOG = logTool("../../data/log/word2vector")

    vecList = []
    labelList = []
    with open(INPUTPATH, "r") as f:
        index = 0
        for line in f:
            indexLength = 0
            ll = line.strip()
            label, vec = ll.split(",")
            vec = vec.split(" ")
            vec = [float(vecItem) for vecItem in vec]
            labelList.append(label)
            vecList.append(vec)
            index = index + 1
            if index % 3000 == 0:
                print(index)

    clf = LogisticRegression()
    # clf = svm.SVC(C=1000)
    clf.fit(vecList, labelList)
    score = clf.score(vecList, labelList)
    print("score_train:%s" % score)

    vecList = []
    labelList = []
    with open(INPUTPATH_TEST, "r") as f:
        index = 0
        for line in f:
            indexLength = 0
            ll = line.strip()
            label, vec = ll.split(",")
            vec = vec.split(" ")
            vec = [float(vecItem) for vecItem in vec]
            labelList.append(label)
            vecList.append(vec)
            index = index + 1
            if index % 3000 == 0:
                print(index)
    score = clf.score(vecList, labelList)
    print("score_test:%s" % score)
