import sys
sys.path.append("../util")
from util import logTool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

INPUTPATH_TRAIN = "../../data/sample_30"
INPUTPATH_TEST = "../../data/baseProcess"
MODELPATH = "../../data/model/MLP_30_tfidf"
LOG = logTool("../../data/log/tfidf_LR")
OUTPUT_RESULT = "../../data/RELEASE_20180727"

LOG.info("Starting")
sentenses_train = []
labels_train = []
LOG.info("Starting loading train data")
with open(INPUTPATH_TRAIN, "r") as f:
    for line in f:
        label, sentense = line.split(",")
        labels_train.append(label)
        sentenses_train.append(sentense)
    print("finish loading train data")
    LOG.info("finish loading train data")

vectorizer = TfidfVectorizer(
    min_df=3, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1)
vectorizer.fit(sentenses_train)
print("finish TFIDF transform")
LOG.info("finish TFIDF transform")

sentenses_test = []
idList = []
LOG.info("Starting loading test data")
with open(INPUTPATH_TEST, "r") as f:
    for line in f:
        ids, sentense = line.split(",")
        idList.append(ids)
        sentenses_test.append(sentense)
    print("finish loading test data")
    LOG.info("finish loading test data")

test = vectorizer.transform(sentenses_test)
print("finish TFIDF transform test")
LOG.info("finish TFIDF transform test")
# print(len(test))

clf = joblib.load('%s.pkl' % MODELPATH)
predict = clf.predict(test)

fout = open("%s.csv" % OUTPUT_RESULT, "w")
fout.write("id,class\n")
for i in range(len(predict)):
    fout.write("%s,%s\n" % (i, predict[i]))

print("completed")
LOG.info("completed")