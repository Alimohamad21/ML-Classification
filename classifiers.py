import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot

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
    # acc_data = []
    for k in range(7,8):  # Tested in range(1, 200)  and 7 was highest accuracy
        clf = KNeighborsClassifier(n_neighbors=k)
        acc_list_temp = crossValidate(X_train, y_train, clf)
        # acc_data.append(acc_list_temp[1])
        if acc_list_temp[1] > hyperParam[2]:
            hyperParam = [k, acc_list_temp[0], acc_list_temp[1]]
    # pyplot.plot(range(5, 205, 5), acc_data)
    # pyplot.title('KNN with K Tuned')
    # pyplot.xlabel("K")
    # pyplot.ylabel("Accuracy")
    # pyplot.show()
    # print(f'Best k = {hyperParam[0]} with best accuracy = {hyperParam[2]}')
    clf = KNeighborsClassifier(n_neighbors=hyperParam[0])
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    reportClassifier(y_test, y_predict)


def adaBoost(X_train, X_test, y_train, y_test):
    hyperParam = [1, 1, 0]
    # acc_data = []
    for n in range(165, 166):  # Tested in range(1, 200)  and 165 was highest accuracy
        clf = AdaBoostClassifier(n_estimators=n)
        acc_list_temp = crossValidate(X_train, y_train, clf)
        # acc_data.append(acc_list_temp[1])
        if acc_list_temp[1] > hyperParam[2]:
            hyperParam = [n, acc_list_temp[0], acc_list_temp[1]]
    # pyplot.plot(range(5, 205, 5), acc_data)
    # pyplot.title('Adaboost with n-estimators')
    # pyplot.xlabel("n estimators")
    # pyplot.ylabel("Accuracy")
    # pyplot.show()
    # print(f'Best n = {hyperParam[0]} with best accuracy = {hyperParam[2]}')
    clf = AdaBoostClassifier(n_estimators=hyperParam[0])
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    reportClassifier(y_test, y_predict)


def randomForest(X_train, X_test, y_train, y_test):
    hyperParam = [1, 1, 0]
    # acc_data = []
    for n in range(75, 76):  # Tested in range(1, 200)  and 75 was highest accuracy
        clf = RandomForestClassifier(n_estimators=n, max_depth=3)
        acc_list_temp = crossValidate(X_train, y_train, clf)
        # acc_data.append(acc_list_temp[1])
        if acc_list_temp[1] > hyperParam[2]:
            hyperParam = [n, acc_list_temp[0], acc_list_temp[1]]
    # pyplot.plot(range(5, 205, 5), acc_data)
    # pyplot.title('Random Forest with n-estimators')
    # pyplot.xlabel("n estimators")
    # pyplot.ylabel("Accuracy")
    # pyplot.show()
    # print(f'Best n = {hyperParam[0]} with best accuracy = {hyperParam[2]}')
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
    print('Report:')
    print(f"{classification_report(y_test, y_pred)}\n", )


def classify():
    dataFrame = readCsv()
    X_train, X_test, y_train, y_test = split(dataFrame)
    y_train = np.ravel(y_train, order='C')  # to avoid warnings
    classifiers = {'DECISION TREE': decisionTree, 'NAIVE BAYES': naiveBayes, 'KNN': knn, 'ADABOOST': adaBoost,
                   'RANDOM FOREST': randomForest}
    for classifierName, classifier in classifiers.items():
        print(f'*************************** CLASSIFICATION USING {classifierName} ***************************\n\n')
        classifier(X_train, X_test, y_train, y_test)
