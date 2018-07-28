import sys
sys.path.append("../util")
from util import logTool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.externals import joblib

INPUTPATH_TRAIN = "../../data/sample_30"
INPUTPATH_TEST = "../../data/sample_10_test"
OUTPUTPATH_TRAIN = "../../data/sample_30_tfidf"
OUTPUTPATH_TEST = "../../data/sample_30_tfidf_test"
MODELPATH = "../../data/model/MLP_30_tfidf"
LOG = logTool("../../data/log/tfidf_MLP")

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

sentenses_test = []
labels_test = []
LOG.info("Starting loading test data")
with open(INPUTPATH_TEST, "r") as f:
    for line in f:
        label, sentense = line.split(",")
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

hidden_layer_sizes = [500]
activation = 'logistic'
max_iter = 200
clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                    learning_rate='adaptive', max_iter=max_iter, verbose=True)
clf.fit(train, labels_train)
print("finish Model Training")
LOG.info("finish Model Training")


predict = clf.predict(test)
LOG.info("Using MLP with hidden_layer_sizes = %s, activation = %s, max_iter = %s" %
         (hidden_layer_sizes, activation, max_iter))

accuracy = accuracy_score(labels_test, predict)
LOG.info("accuracy_score: %s" % (accuracy))
print("accuracy_score: %s" % (accuracy))

confusionMatrix = confusion_matrix(labels_test, predict)
LOG.info("confusion_matrix: \n %s" % (confusionMatrix))
print("confusion_matrix: %s" % (confusionMatrix))

joblib.dump(clf, '%s.pkl' % MODELPATH)
