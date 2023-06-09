"""

    Attributes:
        k_max (int): The maximum number of centroids to be considered within the K-means algorithm performing location encoding.
        k_interval (int): The centroid increasing interval so from k_init to k_max intervals of k_interval are used.
        k_init (int): The initial number of centroids to be considered within the K-means algorithm performing location encoding
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# K-means information (Location encoding)
k_max = 60
k_interval = 2
k_init = 4


def train_kmeans(df: pd.DataFrame):
    location_df = df[['latitude', 'longitude']]

    sil_scores = calculate_optimal_k(location_df)
    k_values = range(k_init, k_max + k_interval, k_interval)

    silhouette_optimal = np.argmax(sil_scores)
    k_value = (k_values[silhouette_optimal])
    print("K-value: ", k_value)

    k_means = KMeans(n_clusters=k_value, n_init=10)
    k_means.fit(location_df)
    return k_means


def calculate_optimal_k(data):
    sil = []

    k = k_init
    while k <= k_max and k < len(data):
        k_means = KMeans(n_clusters=k, n_init=10).fit(data)
        labels = k_means.labels_

        sil.append(silhouette_score(data, labels))
        print(k)
        k += k_interval
    return sil

