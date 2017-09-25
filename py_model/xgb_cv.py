import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from sklearn.model_selection import train_test_split
import os
import pprint


def rmsle(preds, y):
    return np.sqrt(np.mean(np.square(np.log1p(np.exp(preds)) - np.log1p(np.exp(y)))))

if __name__ == '__main__':

    print('Loading train data')
    train = pd.read_pickle('../input/process/train_df.pkl')
    test = pd.read_pickle('../input/process/test_df.pkl')

    f_wo = ['id', 'log_trip_duration', 'pickup_date', 'pickup_datetime',
            'speed', 'trip_duration', 'dropoff_datetime']
    f_to_use = [f for f in list(train) if f not in f_wo]

    y = train.log_trip_duration.values

    Xtr, Xv, ytr, yv = train_test_split(train[f_to_use].values, y, test_size=0.2, random_state=1987)
    dtrain = xgb.DMatrix(Xtr, label=ytr)
    dvalid = xgb.DMatrix(Xv, label=yv)
    dtest = xgb.DMatrix(test[f_to_use].values)
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

    xgb_pars = {'min_child_weight': 50,
                'eta': 0.3,
                'colsample_bytree': 0.3,
                'max_depth': 10,
                'subsample': 0.8,
                'lambda': 1.,
                'nthread': -1,
                'booster': 'gbtree',
                'silent': 1,
                'eval_metric': 'rmse',
                'objective': 'reg:linear'}

    print('xgb training...')
    model = xgb.train(xgb_pars, dtrain, 60, watchlist, early_stopping_rounds=50,
                      maximize=False, verbose_eval=10)

    print('cv', rmsle(model.predict(dvalid), yv))
    ytest = model.predict(dtest)
    test.loc[:, 'trip_duration'] = np.exp(ytest) - 1
    test[['id', 'trip_duration']].to_csv('../output/ny_taxi_822.csv', index=False)



