import pandas as pd
from imblearn.datasets import make_imbalance
from sklearn.model_selection import train_test_split


def readCsv():
    return pd.read_csv('magic04.data')


def split(dataFrame):
    x, y = dataFrame.iloc[:, :-1], dataFrame.iloc[:, [-1]]  # split features and label
    x, y = make_imbalance(x, y, sampling_strategy={'g': 6688, 'h': 6688}, random_state=14)
    return train_test_split(x, y, test_size=0.3, train_size=0.7, random_state=17)

