import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import ensemble
from scipy.stats import ks_2samp
from sklearn import random_projection
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.metrics import mean_squared_error

read_test = True
num_decimals = 32
num_features = 1000
threshold_p_value = 0.01
threshold_static = 0.3
num_folds = 5

train = pd.read_csv('train.csv')

y_train = train['target']
y_train = np.log1p(y_train)

train.drop("ID", axis=1, inplace=True)
train.drop("target", axis=1, inplace=True)

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

ntrain = len(train)
weight = ((train != 0).sum() / len(train)).values
ntest = len(test)
tmp = pd.concat([train, test])  # RandomProjection
tmp_train = train[train != 0]
tmp_test = test[test != 0]
train["weight_count"] = (tmp_train * weight).sum(axis=1)
test["weight_count"] = (tmp_test * weight).sum(axis=1)
train["count_not0"] = (train != 0).sum(axis=1)
test["count_not0"] = (test != 0).sum(axis=1)
train["sum"] = train.sum(axis=1)
test["sum"] = test.sum(axis=1)
train["var"] = tmp_train.var(axis=1)
test["var"] = tmp_test.var(axis=1)
train["median"] = tmp_train.median(axis=1)
test["median"] = tmp_test.median(axis=1)
train["mean"] = tmp_train.mean(axis=1)
test["mean"] = tmp_test.mean(axis=1)
train["std"] = tmp_train.std(axis=1)
test["std"] = tmp_test.std(axis=1)
train["max"] = tmp_train.max(axis=1)
test["max"] = tmp_test.max(axis=1)
train["min"] = tmp_train.min(axis=1)
test["min"] = tmp_test.min(axis=1)
train["skew"] = tmp_train.skew(axis=1)
test["skew"] = tmp_test.skew(axis=1)
train["kurtosis"] = tmp_train.kurtosis(axis=1)
test["kurtosis"] = tmp_test.kurtosis(axis=1)
del (tmp_train)
del (tmp_test)
NUM_OF_COM = 100  # need tuned
transformer = random_projection.SparseRandomProjection(n_components=NUM_OF_COM)
RP = transformer.fit_transform(tmp)
rp = pd.DataFrame(RP)
columns = ["RandomProjection{}".format(i) for i in range(NUM_OF_COM)]
rp.columns = columns

rp_train = rp[:ntrain]
rp_test = rp[ntrain:]
rp_test.index = test.index

# concat RandomProjection and raw data
train = pd.concat([train, rp_train], axis=1)
test = pd.concat([test, rp_test], axis=1)

del (rp_train)
del (rp_test)
print(train.shape)


def rmsle_cv(model):
    kf = KFold(num_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train, y_train, scoring="neg_mean_squared_error", cv=kf))
    return rmse


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.modesls = models

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        for model in self.models_:
            model.fit(X, y)

    def predixt(self, X):
        predictions = np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions, axis=1)
