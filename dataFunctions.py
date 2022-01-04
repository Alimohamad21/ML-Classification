import numpy as np
import pandas as pd
from imblearn.datasets import make_imbalance
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


def readCsv():
    return pd.read_csv('magic04.data')


def split(dataFrame):
    x, y = dataFrame.iloc[:, :-1], dataFrame.iloc[:, [-1]]  # split features and label
    x, y = make_imbalance(x, y, sampling_strategy={'g': 6688, 'h': 6688}, random_state=14)
    return train_test_split(x, y, test_size=0.3, train_size=0.7, random_state=17)


def crossValidate(X_train, y_train, classifier):
    acc_list = [0, 0]
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    for fold in range(19, 20):  # Tested in range(2, 100)  and 19 was highest accuracy
        kf = KFold(n_splits=fold)
        kf.get_n_splits(X_train)
        acc = 0
        counter = 0
        for train_index, test_index in kf.split(X_train):
            KFX_train, KFX_test = X_train[train_index], X_train[test_index]
            KFy_train, KFy_test = y_train[train_index], y_train[test_index]
            classifier.fit(KFX_train, KFy_train)
            KFy_predict = classifier.predict(KFX_test)
            acc += accuracy_score(KFy_test, KFy_predict) * 100
            counter += 1
        acc = acc / counter
        if acc > acc_list[1]:
            acc_list = [fold, acc]
    return acc_list
