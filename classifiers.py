from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import numpy as np
from sklearn.model_selection import KFold

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
    best_acc = [1, 1, 0]
    for k in range(1, 20):
        clf = KNeighborsClassifier(n_neighbors=k)
        acc_list_temp = crossValidate(X_train, y_train, clf)
        if acc_list_temp[1] > best_acc[2]:
            best_acc = [k, acc_list_temp[0], acc_list_temp[1]]
    print(best_acc)

    # y_predict = clf.predict(X_test)
    # cal_accuracy(y_test, y_predict)
def crossValidate(X_train, y_train, classifier):
    acc_list = [0, 0]
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    for i in range(2, 20):
        kf = KFold(n_splits=i)
        print(kf)
        kf.get_n_splits(X_train)
        acc = 0
        counter = 0
        for train_index, test_index in kf.split(X_train):
            print("TRAIN:", train_index, "TEST:", test_index)
            KFX_train, KFX_test = X_train[train_index], X_train[test_index]
            KFy_train, KFy_test = y_train[train_index], y_train[test_index]
            classifier.fit(KFX_train, KFy_train)
            KFy_predict = classifier.predict(KFX_test)
            acc += accuracy_score(KFy_test, KFy_predict) * 100
            counter += 1
        acc = acc/counter
        if acc > acc_list[1]:
            acc_list = [i, acc]
    return acc_list


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
# naiveBayes(X_train, X_test, y_train, y_test)
knn(X_train, X_test, y_train, y_test)