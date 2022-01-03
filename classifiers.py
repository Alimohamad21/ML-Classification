from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from dataFunctions import readCsv, split, crossValidate


def decisionTree(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier().fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    reportClassifier(y_test, y_predict)


def naiveBayes(X_train, X_test, y_train, y_test):
    clf = GaussianNB().fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    reportClassifier(y_test, y_predict)


def knn(X_train, X_test, y_train, y_test):
    hyperParam = [1, 1, 0]
    for k in range(7, 8):  # Tested in range(1, 100)  and 7 was highest accuracy
        clf = KNeighborsClassifier(n_neighbors=k)
        acc_list_temp = crossValidate(X_train, y_train, clf)
        if acc_list_temp[1] > hyperParam[2]:
            hyperParam = [k, acc_list_temp[0], acc_list_temp[1]]
    print(hyperParam)
    clf = KNeighborsClassifier(n_neighbors=hyperParam[0])
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    reportClassifier(y_test, y_predict)


def adaBoost(X_train, X_test, y_train, y_test):
    hyperParam = [1, 1, 0]
    for n in range(94, 95):  # Tested in range(1, 100)  and 94 was highest accuracy
        clf = AdaBoostClassifier(n_estimators=n)
        acc_list_temp = crossValidate(X_train, y_train, clf)
        if acc_list_temp[1] > hyperParam[2]:
            hyperParam = [n, acc_list_temp[0], acc_list_temp[1]]
    print(hyperParam)
    clf = AdaBoostClassifier(n_estimators=hyperParam[0])
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    reportClassifier(y_test, y_predict)


def randomForest(X_train, X_test, y_train, y_test):
    hyperParam = [1, 1, 0]
    for n in range(37, 38):  # Tested in range(1, 100)  and 37 was highest accuracy
        clf = RandomForestClassifier(n_estimators=n, max_depth=3)
        acc_list_temp = crossValidate(X_train, y_train, clf)
        if acc_list_temp[1] > hyperParam[2]:
            hyperParam = [n, acc_list_temp[0], acc_list_temp[1]]
    print(hyperParam)
    clf = RandomForestClassifier(max_depth=hyperParam[0])
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    reportClassifier(y_test, y_predict)





def reportClassifier(y_test, y_pred):
    TG, FG = confusion_matrix(y_test, y_pred)[0]
    FH, TH = confusion_matrix(y_test, y_pred)[1]
    print("Confusion Matrix:\n")
    print(f'TG:{TG}\tFG:{FG}\nFH:{FH}\tTH:{TH}\n')
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100}\n")
    print(f"Report: {classification_report(y_test, y_pred)}\n", )


def classify():
    dataFrame = readCsv()
    X_train, X_test, y_train, y_test = split(dataFrame)
    classifiers = [decisionTree, naiveBayes, knn, adaBoost, randomForest]
    for classifier in classifiers:
        classifier(X_train, X_test, y_train, y_test)
