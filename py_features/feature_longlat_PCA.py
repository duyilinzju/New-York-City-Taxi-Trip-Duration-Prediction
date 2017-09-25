from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import pandas as pd
import os


if __name__ == '__main__':

    print('Get long-lat PCA features')

    use_cols = ['id', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']

    train = pd.read_pickle('../input/process/train.pkl')
    train = train[use_cols]

    test = pd.read_csv('../input/nyc-taxi-trip-duration/test.csv',
                          usecols=use_cols)

    coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                        train[['dropoff_latitude', 'dropoff_longitude']].values,
                        test[['pickup_latitude', 'pickup_longitude']].values,
                        test[['dropoff_latitude', 'dropoff_longitude']].values))

    pca_df = pd.concat([train, test])

    # 100 Clusters for different location
    sample_ind = np.random.permutation(len(coords))[:500000]
    kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])

    pca_df.loc[:, 'pickup_cluster'] = kmeans.predict(pca_df[['pickup_latitude', 'pickup_longitude']])
    pca_df.loc[:, 'dropoff_cluster'] = kmeans.predict(pca_df[['dropoff_latitude', 'dropoff_longitude']])

    # PCA transformation for longitude and latitude
    pca = PCA().fit(coords)
    pca_df['pickup_pca0'] = pca.transform(pca_df[['pickup_latitude', 'pickup_longitude']])[:, 0]
    pca_df['pickup_pca1'] = pca.transform(pca_df[['pickup_latitude', 'pickup_longitude']])[:, 1]
    pca_df['dropoff_pca0'] = pca.transform(pca_df[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
    pca_df['dropoff_pca1'] = pca.transform(pca_df[['dropoff_latitude', 'dropoff_longitude']])[:, 1]

    # keep the newly-generated long-lat features
    pca_df.drop(['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'],
               inplace=True, axis=1)

    if not os.path.isdir('../input/process'):
        os.makedirs('../input/process')

    print('Save long-lat pca features')
    pca_df.to_pickle('../input/process/longlat_PCA.pkl')
