import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import ensemble
from scipy.stats import ks_2samp

read_test = True
num_decimals = 32
num_features = 1000
threshold_p_value = 0.01
threshold_static = 0.3

train = pd.read_csv('train.csv')

y_train = train['target']
y_train = np.log1p(y_train)

train.drop("ID", axis=1, inplace=True)

cols_with_onlyone_val = train.columns[train.nunique() == 1]

train.drop(cols_with_onlyone_val, axis=1, inplace=True)

train = train.round(num_decimals)

global test
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

col = pd.DataFrame({'importance': model.feature_importances_, 'feature': train.columns}).sort_values(by='importance',
                                                                                                     ascending=[False])[
      :num_features]['feature'].values

train = train[col]

if read_test:
    test = test[col]

print(train.shape)

if read_test:
    diff_cols = []
    for col in train.columns:
        statistic, pvalue = ks_2samp(train[col].values, test[col].values)
        if pvalue <= threshold_p_value and np.abs(statistic) > threshold_static:
            diff_cols.append(col)
    for col in diff_cols:
        if col in train.columns:
            train.drop(col, axis=1, inplace=True)
            test.drop(col, axis=1, inplace=True)
    print(train.shape)
