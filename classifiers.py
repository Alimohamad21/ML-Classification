import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

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
    hyperParam = [1, 1, 0]
    for k in range(1, 100):  # Tested in range(1, 100)  and 7 was highest accuracy
        clf = KNeighborsClassifier(n_neighbors=k)
        acc_list_temp = crossValidate(X_train, y_train, clf)
        if acc_list_temp[1] > hyperParam[2]:
            hyperParam = [k, acc_list_temp[0], acc_list_temp[1]]
    print(hyperParam)
    clf = KNeighborsClassifier(n_neighbors=hyperParam[0])
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    cal_accuracy(y_test, y_predict)

def adaBoost(X_train, X_test, y_train, y_test):
    hyperParam = [1, 1, 0]
    for n in range(1, 100):  # Tested in range(1, 100)  and 7 was highest accuracy
        print(n)
        clf = AdaBoostClassifier(n_estimators=n)
        acc_list_temp = crossValidate(X_train, y_train, clf)
        if acc_list_temp[1] > hyperParam[2]:
            hyperParam = [n, acc_list_temp[0], acc_list_temp[1]]
    print(hyperParam)
    clf = AdaBoostClassifier(n_estimators=hyperParam[0])
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    cal_accuracy(y_test, y_predict)

def randomForest(X_train, X_test, y_train, y_test):
    hyperParam = [1, 1, 0]
    for d in range(1, 100):  # Tested in range(1, 100)  and 7 was highest accuracy
        clf = RandomForestClassifier(max_depth=d)
        acc_list_temp = crossValidate(X_train, y_train, clf)
        if acc_list_temp[1] > hyperParam[2]:
            hyperParam = [d, acc_list_temp[0], acc_list_temp[1]]
    print(hyperParam)
    clf = RandomForestClassifier(max_depth=hyperParam[0])
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    cal_accuracy(y_test, y_predict)

def crossValidate(X_train, y_train, classifier):
    acc_list = [0, 0]
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    for fold in range(19, 20):  # Tested in range(1, 100)  and 19 was highest accuracy
        kf = KFold(n_splits=fold)
        kf.get_n_splits(X_train)
        acc = 0
        counter = 0
        for train_index, test_index in kf.split(X_train):
            KFX_train, KFX_test = X_train[train_index], X_train[test_index]
            KFy_train, KFy_test = y_train[train_index], y_train[test_index]
            classifier.fit(KFX_train, np.ravel(KFy_train, order='C'))
            KFy_predict = classifier.predict(KFX_test)
            acc += accuracy_score(KFy_test, KFy_predict) * 100
            counter += 1
        acc = acc / counter
        if acc > acc_list[1]:
            acc_list = [fold, acc]
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
# knn(X_train, X_test, y_train, y_test)
adaBoost(X_train, X_test, y_train, y_test)