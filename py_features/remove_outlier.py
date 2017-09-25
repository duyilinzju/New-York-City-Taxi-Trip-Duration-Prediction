import os
import pandas as pd
import numpy as np


def remove_outliers(df, feature, low_limit=0.005, high_limit=0.995):

    min = df[feature].quantile(low_limit)
    max = df[feature].quantile(high_limit)

    return df.loc[(df[feature] < max) & (df[feature] > min)]


if __name__ == '__main__':

    print('Load train data')

    train = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv',
                        parse_dates=['pickup_datetime', 'dropoff_datetime'])

    print('Total rows in training data: ', train.shape[0])

    train.loc[:, 'log_trip_duration'] = np.log1p(train['trip_duration'].values)

    # remove outliers
    # train = remove_outliers(train, 'log_trip_duration')

    print('Total rows after removing outliers: ', train.shape[0])

    if not os.path.isdir('../input/process'):
        os.makedirs('../input/process')

    print('Save processed train df')
    train.to_pickle('../input/process/train.pkl')

