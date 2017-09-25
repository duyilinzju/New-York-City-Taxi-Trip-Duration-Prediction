import pandas as pd
import os

if __name__ == '__main__':

    print('Get osrm features')

    osrm_test = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_test.csv')
    osrm_train1 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv')
    osrm_train2 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_2.csv')

    cols_keep = ['id', 'total_distance', 'total_travel_time', 'number_of_steps']
    osrm_df = pd.concat([osrm_test, osrm_train1, osrm_train2])[cols_keep]

    if not os.path.isdir('../input/process'):
        os.makedirs('../input/process')

    print('Save osrm features')
    osrm_df.to_pickle('../input/process/osrm.pkl')

