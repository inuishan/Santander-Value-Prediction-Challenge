import numpy as np
import pandas as pd


read_test = False

train = pd.read_csv('train.csv')

y_train = train['target']
y_train = np.log1p(y_train)

train.drop("ID", axis=1, inplace=True)

cols_with_onlyone_val = train.columns[train.nunique() == 1]

train.drop(cols_with_onlyone_val, axis=1, inplace=True)


if read_test:
    test = pd.read_csv('test.csv')
    test_ID= test['ID']
    test.drop('ID', axis=1, inplace=True)
    test.drop(cols_with_onlyone_val, axis=1, inplace=True)
