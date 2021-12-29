from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from dataFunctions import readCsv, split


def decisionTree(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier().fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    cal_accuracy(y_test, y_predict)


def naiveBayes(X_train, X_test, y_train, y_test):
    clf = GaussianNB().fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    cal_accuracy(y_test, y_predict)


def knn(X_train, X_test, y_train, y_test):
    tunedK = 5
    clf = KNeighborsClassifier(n_neighbors=tunedK).fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    cal_accuracy(y_test, y_predict)


def cal_accuracy(y_test, y_pred):
    TG, FG = confusion_matrix(y_test, y_pred)[0]
    FH, TH = confusion_matrix(y_test, y_pred)[1]
    print("Confusion Matrix:\n")
    print(f'TG:{TG}\tFG:{FG}\nFH:{FH}\tTH:{TH}\n')
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100}\n")
    print(f"Report: {classification_report(y_test, y_pred)}\n", )


dataFrame = readCsv()
X_train, X_test, y_train, y_test = split(dataFrame)
# decisionTree(X_train, X_test, y_train, y_test)
naiveBayes(X_train, X_test, y_train, y_test)
