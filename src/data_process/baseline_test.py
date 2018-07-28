import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

print("hello world")
train_head = []
train_class_head = []
test_head = []
test_class_head = []
with open("../../data/train_set.csv") as f:
    cal_1 = 10000
    cal_2 = 20000
    index = 0
    for line in f:
        if index == 0:
            index = index + 1
            continue
        ll = line.split(",")
        if index <= cal_1:
            train_class_head.append(ll[3])
            train_head.append(ll[2])
        elif index > cal_1 and index <= cal_2:
            test_class_head.append(ll[3])
            test_head.append(ll[2])

        index = index + 1

print(len(train_head))
print(len(test_head))
vectorizer = TfidfVectorizer(
    min_df=3, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1)
vectorizer.fit(train_head)
train = vectorizer.transform(train_head)
test = vectorizer.transform(test_head)
clf = LogisticRegression(C=2)
clf.fit(train, train_class_head)
predict = clf.predict(test)
print(len(predict))
print(len(test_class_head))
print(accuracy_score(test_class_head, predict))
print(confusion_matrix(test_class_head, predict))
