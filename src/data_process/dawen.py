import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression,LassoCV
from sklearn.svm import LinearSVC

training=pd.read_csv("C:\\Users\\dell\\Documents\\python3\\newdata.csv")
train_head=training.head(20000)['word_seg']
class_head=training.head(20000)['class']
vectorizer=TfidfVectorizer( min_df=3,max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
train=train_head[0:10000]
test=train_head[10000:20000]
vectorizer.fit(train)
train=vectorizer.transform(train)
test=vectorizer.transform(test)
clf = LogisticRegression(C=4)
clf1=LinearSVC()
clf.fit(train,class_head[0:10000])
predict=clf.predict(test)
print(accuracy_score(class_head[10000:20000],predict))
print(confusion_matrix(class_head[10000:20000],predict))