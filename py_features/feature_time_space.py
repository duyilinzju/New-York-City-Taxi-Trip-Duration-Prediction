from datetime import timedelta
import os
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy as np

# https://www.kaggle.com/the1owl/easy-as-baking-a-cake-if-baking-a-cake-is-easy
cal = USFederalHolidayCalendar()
holidays = [d.date() for d in cal.holidays(start='2016-01-01', end='2016-06-30').to_pydatetime()]
holidays_prev = [d + timedelta(days=-1) for d in holidays]
holidays_after = [d + timedelta(days=1) for d in holidays]

def get_datetime_feature(df):
    df.loc[:, 'pickup_weekday'] = df['pickup_datetime'].dt.weekday
    df.loc[:, 'pickup_hour_weekofyear'] = df['pickup_datetime'].dt.weekofyear
    df.loc[:, 'pickup_hour'] = df['pickup_datetime'].dt.hour
    df.loc[:, 'pickup_minute'] = df['pickup_datetime'].dt.minute
    df.loc[:, 'pickup_week_hour'] = df['pickup_weekday'] * 24 + df['pickup_hour']
    df.loc[:, 'pickup_date'] = df['pickup_datetime'].dt.date
    df.loc[:, 'holiday_after'] = df.pickup_date.map(lambda x: 1 if x in holidays_after else 0)
    df.loc[:, 'holiday_prev'] = df.pickup_date.map(lambda x: 1 if x in holidays_prev else 0)
    df.loc[:, 'holiday'] = df.pickup_date.map(lambda x: 1 if x in holidays else 0)

    return df

# ref: https://www.kaggle.com/gaborfodor/from-eda-to-the-top-lb-0-367
def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

def bearing_array(lat1, lng1, lat2, lng2):
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

if __name__ == '__main__':

    print('Get datetime features')

    # train = pd.read_pickle('../input/process/train.pkl')
    train = pd.read_csv('../input/nyc-taxi-trip-duration/train.csv',
                        parse_dates=['pickup_datetime', 'dropoff_datetime'])
    train.loc[:, 'log_trip_duration'] = np.log1p(train['trip_duration'].values)

    test = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv',
                          parse_dates=['pickup_datetime'])

    train = get_datetime_feature(train)
    test = get_datetime_feature(test)

    print('Get distance features')

    train.loc[:, 'distance_haversine'] = haversine_array(train['pickup_latitude'].values,
                                                         train['pickup_longitude'].values,
                                                         train['dropoff_latitude'].values,
                                                         train['dropoff_longitude'].values)
    train.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(train['pickup_latitude'].values,
                                                                        train['pickup_longitude'].values,
                                                                        train['dropoff_latitude'].values,
                                                                        train['dropoff_longitude'].values)
    train.loc[:, 'direction'] = bearing_array(train['pickup_latitude'].values, train['pickup_longitude'].values,
                                              train['dropoff_latitude'].values, train['dropoff_longitude'].values)

    test.loc[:, 'distance_haversine'] = haversine_array(test['pickup_latitude'].values, test['pickup_longitude'].values,
                                                        test['dropoff_latitude'].values,
                                                        test['dropoff_longitude'].values)
    test.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(test['pickup_latitude'].values,
                                                                       test['pickup_longitude'].values,
                                                                       test['dropoff_latitude'].values,
                                                                       test['dropoff_longitude'].values)
    test.loc[:, 'direction'] = bearing_array(test['pickup_latitude'].values, test['pickup_longitude'].values,
                                             test['dropoff_latitude'].values, test['dropoff_longitude'].values)

    train.loc[:, 'speed'] = train.distance_dummy_manhattan / train.trip_duration

    if not os.path.isdir('../input/process'):
        os.makedirs('../input/process')

    print('Save datetime and distance features')
    train.to_pickle('../input/process/train_time_space.pkl')
    test.to_pickle('../input/process/test_time_space.pkl')

#    [ id     pickup_datetime  pickup_weekday  pickup_hour_weekofyear
#      pickup_hour  pickup_minute  pickup_week_hour pickup_date  holiday_after
#      holiday_prev  holiday]