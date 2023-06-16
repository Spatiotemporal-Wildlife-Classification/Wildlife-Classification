"""This file automates the encoding of observation location metadata features through the use of K-means clustering

    In order to automate the selection of the number of clusters to best describe the locations of the provided data,
    a Silhouette score is created across a set range of cluster values. The global maximum silhouette score is selected as the
    optimal number of centroids best fitting the data.
    The Silhouette score calculates the mean intra-cluster distance and the mean nearest-cluster distance.
    For more information on the Silhouette score please read the documentation provided by sklearn here: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html

    This file, provides methods to handle the automated selection of the number of centroids and model saving.
    Attributes:
        k_max (int): The maximum number of centroids to be considered within the K-means algorithm performing location encoding.
        k_interval (int): The centroid increasing interval so from k_init to k_max intervals of k_interval are used.
        k_init (int): The initial number of centroids to be considered within the K-means algorithm performing location encoding
        save_path (str): The directory in which K-means models are saved. The `models/k_clusters/` directory.
"""

# General
import pickle
import numpy as np
import pandas as pd

# Modelling
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Project
from src.models.meta import pipelines

# K-means centroid range
k_max = 60
k_interval = 2
k_init = 4

# Model save path
save_path = '/models/k_clusters/'


def silhouette_process(df: pd.DataFrame, validation_file: str):
    """This method defines the entire process of automating the centroid selection and returning the model created k-means model

    Args:
        df (DataFrame): The dataframe containing each observation and its associated dataframe, including the latitude and longitude location.
        validation_file (str): The name of the validation dataset file. This filename will be augmented to provide a name for the saved K-means model.

    Returns:
        (K-means model): Returns a trained K-means model with the optimal number of centroids determined by the Silhouette score.
    """
    k_means = train_kmeans(df)  #Train the K-means algorithm across the range of centroids and determine optimal centroids
    model_name = validation_file[:-14]  # Extract model name from the validation name
    save_k_means(k_means, model_name)  # Save the optimal model
    return k_means


def train_kmeans(df: pd.DataFrame):
    """Method trains the K-means models across the range of centroids and determines the optimal centroids using the Silhouette score.

    For more information on how the Silhouette score works, please review `notebooks/meta_modelling/location_feature_k_means_automation` notebook.

    Args:
        df (DataFrame): The dataframe containing all observation features including the latitude and longitude location.

    Returns:
        (K-means model): The optimal trained K-means model as determined by the global maximum Silhouette score
    """
    location_df = df[['latitude', 'longitude']]  # Extract longitude and latidues

    sil_scores = calculate_optimal_k(location_df)  # Perform Silhouette score calculations
    k_values = range(k_init, k_max + k_interval, k_interval)  # Generate the centroid range

    silhouette_optimal = np.argmax(sil_scores)  # Extract the optimal number of centroids (this is the index value)
    k_value = (k_values[silhouette_optimal])  # Extract the number of centroids from the range using the index

    k_means = KMeans(n_clusters=k_value, n_init=10)  # Recreate K-means model
    k_means.fit(location_df)  # Train using k centroids
    return k_means


def calculate_optimal_k(data):
    """This method performs the Silhouette score calculations, creating a list of Silhouette scores for the range of centroids

    Args:
        data (DataFrame): A  dataframe containing the latitude and longitude columns.

    Returns:
        A list of Silhouette scores aligned with the range of centroids.
    """
    sil = []

    k = k_init
    while k <= k_max and k < len(data):
        k_means = KMeans(n_clusters=k, n_init=10).fit(data)  # Train model with k centroids
        labels = k_means.labels_

        sil.append(silhouette_score(data, labels))  # Append model's Silhouette score to the list
        k += k_interval  # Increase k by the interval
    return sil


def save_k_means(k_means: KMeans, model_name: str):
    """Method saves the provided K-means model according to the Pipeline save path.
    
    Args:
        k_means (KMeans): The trained K-means model to be saved.
        model_name (str): The name of the file in which the model will be saved.
    """
    pickle.dump(k_means, open(pipelines.root_path + save_path + model_name + 'k_means.sav', 'wb'))

