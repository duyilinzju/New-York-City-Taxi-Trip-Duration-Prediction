from datetime import timedelta
import os
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy as np

# generate speed related features

if __name__ == '__main__':

    print('Get speed agg features')
    train = pd.read_pickle('../input/process/train_time_space.pkl')
    test = pd.read_pickle('../input/process/test_time_space.pkl')
    longlat_df = pd.read_pickle('../input/process/longlat_PCA.pkl')

    print(longlat_df.head())
    train = train.merge(longlat_df, how='left', on='id')
    test = test.merge(longlat_df, how='left', on='id')

    for gby_col in ['vendor_id', 'pickup_cluster', 'dropoff_cluster', 'pickup_hour', \
                    'pickup_week_hour', 'pickup_date']:
        gby = train.groupby(gby_col)['speed', 'log_trip_duration'].mean()
        gby.columns = ['%s_gby_%s' % (col, gby_col) for col in gby.columns]
        train = train.merge(gby, how='left', left_on=gby_col, right_index=True)
        test = test.merge(gby, how='left', left_on=gby_col, right_index=True)

    for gby_cols in [['pickup_hour', 'pickup_cluster'],
                     ['pickup_hour', 'dropoff_cluster'],
                     ['pickup_cluster', 'dropoff_cluster']]:
        gbys = train.groupby(gby_cols)['speed', 'log_trip_duration'].mean()
        gbys.columns = ['%s_gbys_%s' % (col, '_'.join(gby_cols)) for col in gbys.columns]
        gbys = gbys.reset_index()
        train = train.merge(gbys, how='left', on=gby_cols)
        test = test.merge(gbys, how='left', on=gby_cols)

    if not os.path.isdir('../input/process'):
        os.makedirs('../input/process')

    print('Save speed aggregation features')
    train.to_pickle('../input/process/train_speed_agg.pkl')
    test.to_pickle('../input/process/test_speed_agg.pkl')
