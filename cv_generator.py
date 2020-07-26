#cv generator for time based client data
#choose between holdout 70/15/15 vs cv 85/15 via sklearn.model_selection.learning_curve
import pandas as pd
import numpy as np
from itertools import product
from sklearn import linear_model as lm
from sklearn.model_selection import cross_val_score, cross_validate
N_SPLITS = 10

dates = pd.date_range('2019-07-01', '2020-06-01', freq='MS')
ids = range(99)

df = pd.DataFrame(product(dates, ids))
df['a'] =  np.random.lognormal(0,1,len(df))
df['b'] =  np.random.randint(1,6,len(df)).astype(object)
df['c'] =  np.random.randint(10,16,len(df)).astype(object)
df['y'] =  np.random.lognormal(0.5,1,len(df))
df.rename(columns={0:'date', 1:'id'}, inplace = True)

X = df.copy()
y = X['y']
X.drop('y', axis=1, inplace=True)

def split_time(df):
    assert 'id' in df.columns
    unique_ids = df['id'].unique()
    for i, chunk in enumerate(np.array_split(unique_ids, N_SPLITS)):
        test_ix = df.index[df.id.isin(chunk)]
        train_ix = df.index[~df.id.isin(chunk)]
        assert set(df['id'][df.id.isin(chunk)].unique()) & set(df['id'][~df.id.isin(chunk)].unique()) == set()
        assert set(test_ix) & set(train_ix) == set()
        yield train_ix, test_ix

lr = lm.LinearRegression()
cross_val_score(lr, X[['a','b','c']], y, cv = split_time(X))
cross_validate(lr, X[['a','b','c']], y, cv = split_time(X))