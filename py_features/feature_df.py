from datetime import timedelta
import os
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy as np

# generate speed related features

if __name__ == '__main__':

    print('Merge weather and osrm features')
    train = pd.read_pickle('../input/process/train_speed_agg.pkl')
    test = pd.read_pickle('../input/process/test_speed_agg.pkl')
    weather = pd.read_pickle('../input/process/weather.pkl')
    osrm = pd.read_pickle('../input/process/osrm.pkl')

    df = pd.concat([train, test])

    df = df.merge(weather, on=['pickup_date', 'pickup_hour'], how='left')
    df = df.merge(osrm, on='id', how='left')

    for gby_col in ['pickup_cluster', 'dropoff_cluster', 'pickup_hour', \
                    'pickup_week_hour', 'pickup_date']:
        gby = df.groupby(gby_col)['id'].count().to_frame()  # count the total trip based on time and location
        gby.columns = ['%s_gby_%s' % ('total_trips', gby_col) for col in gby.columns]
        df = df.merge(gby, how='left', left_on=gby_col, right_index=True)

    df['store_and_fwd_flag'] = 1 * (df.store_and_fwd_flag.values == 'Y')

    train = df[:train.shape[0]]
    test = df[train.shape[0]:]

    if not os.path.isdir('../input/process'):
        os.makedirs('../input/process')

    print('Save features_df')
    train.to_pickle('../input/process/train_df.pkl')
    test.to_pickle('../input/process/test_df.pkl')