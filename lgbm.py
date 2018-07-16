import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import ensemble

read_test = False
num_decimals = 32
num_features = 1000

train = pd.read_csv('train.csv')

y_train = train['target']
y_train = np.log1p(y_train)

train.drop("ID", axis=1, inplace=True)

cols_with_onlyone_val = train.columns[train.nunique() == 1]

train.drop(cols_with_onlyone_val, axis=1, inplace=True)

train = train.round(num_decimals)

if read_test:
    test = pd.read_csv('test.csv')
    test_ID = test['ID']
    test.drop('ID', axis=1, inplace=True)
    test.drop(cols_with_onlyone_val, axis=1, inplace=True)
    test = test.round(num_decimals)


def rmsle(y, pred):
    return np.sqrt(np.mean(np.power(y - pred, 2)))


x1, x2, y1, y2 = model_selection.train_test_split(train, y_train.values, test_size=0.2, random_state=7)
model = ensemble.RandomForestRegressor(n_jobs=1, random_state=7, n_estimators=10)
model.fit(x1, y1)


print(rmsle(y2, model.predict(x2)))




