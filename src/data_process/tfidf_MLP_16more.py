import sys
sys.path.append("../util")
from util import logTool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from util import logTool
import random

INPUTPATH_TRAIN = "../../data/sample_10"
INPUTPATH_TEST = "../../data/sample_10_test"
OUTPUTPATH_TRAIN = "../../data/sample_10_tfidf"
OUTPUTPATH_TEST = "../../data/sample_10_tfidf_test"
LOG = logTool("../../data/log/tfidf_16more")

LOG.info("Starting")
sentenses_train = []
labels_train = []
LOG.info("Starting loading train data")
cal = 0
with open(INPUTPATH_TRAIN, "r") as f:
    for line in f:
        label, sentense = line.split(",")
        if label == "16":
            label = "1"
        else:
            if random.random() <= 0.03:
                label = "0"
                cal = cal + 1
            else:
                continue
        labels_train.append(label)
        sentenses_train.append(sentense)
    print(cal)
    print("finish loading train data")
    LOG.info("finish loading train data")

sentenses_test = []
labels_test = []
LOG.info("Starting loading test data")
with open(INPUTPATH_TEST, "r") as f:
    for line in f:
        label, sentense = line.split(",")
        if label == "16":
            label = "1"
        else:
            label = "0"
        labels_test.append(label)
        sentenses_test.append(sentense)
    print("finish loading test data")
    LOG.info("finish loading test data")

vectorizer = TfidfVectorizer(
    min_df=3, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1)
vectorizer.fit(sentenses_train)

train = vectorizer.transform(sentenses_train)
test = vectorizer.transform(sentenses_test)
print("finish TFIDF transform")
LOG.info("finish TFIDF transform")

hidden_layer_sizes = [800, 300, 50]
activation = 'logistic'
max_iter = 1000
LOG.info("Using MLP with hidden_layer_sizes = %s, activation = %s, max_iter = %s" %
         (hidden_layer_sizes, activation, max_iter))
print("Using MLP with hidden_layer_sizes = %s, activation = %s, max_iter = %s" %
      (hidden_layer_sizes, activation, max_iter))
clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                    learning_rate='adaptive', max_iter=max_iter, verbose=True, tol=1e-8)
clf.fit(train, labels_train)
print("finish Model Training")
LOG.info("finish Model Training")


predict = clf.predict(test)

accuracy = accuracy_score(labels_test, predict)
LOG.info("accuracy_score: %s" % (accuracy))
print("accuracy_score: %s" % (accuracy))

confusionMatrix = confusion_matrix(labels_test, predict)
LOG.info("confusion_matrix: \n %s" % (confusionMatrix))
print("confusion_matrix: %s \n" % (confusionMatrix))
