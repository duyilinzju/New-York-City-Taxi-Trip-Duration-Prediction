import os
import pandas as pd

if __name__ == '__main__':

    print('Load weather data')
    weather = pd.read_csv('../input/new-york-city-taxi-trip-hourly-weather-data/Weather.csv',
                          usecols=['pickup_datetime', 'tempi', 'wspdi', 'visi', 'rain', 'snow'],
                          parse_dates=['pickup_datetime'])

    weather.loc[:, 'pickup_date'] = weather['pickup_datetime'].dt.date
    weather.loc[:, 'pickup_hour'] = weather['pickup_datetime'].dt.hour

    # get hourly data and fill nan with nearest non-NaN data
    print('Get hourly weather data')
    weather_hourly = weather.groupby(['pickup_date', 'pickup_hour'])['tempi', 'wspdi', 'visi', 'rain', 'snow'] \
                                      .mean().round(decimals=1).reset_index().interpolate(method='nearest')

    weather_hourly.drop_duplicates(subset=['pickup_date', 'pickup_hour'], inplace=True)

    if not os.path.isdir('../input/process'):
        os.makedirs('../input/process')

    print('Save processed weather df')
    weather_hourly.to_pickle('../input/process/weather.pkl')

