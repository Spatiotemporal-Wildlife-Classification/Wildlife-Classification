import pickle

import pandas as pd
import numpy as np
import tensorflow as tf

import pytz

from src.models.meta.pipelines import elevation_clean, day_night_calculation, season_calc, ohe_season, \
    sub_species_detection
from src.structure.Config import root_dir

import os

data_path = root_dir() + '/data/processed/final_test_observations.csv'
image_path = root_dir() + '/data/final_images/'
model_path = root_dir() + '/models/'
image_model_path = model_path + 'image/'
meta_model_path = model_path + 'meta/'
cluster_model_path = model_path + 'k_clusters/'

# Load base image classifier
base_image_classifier_path = image_model_path + 'family_taxon_classifier'
base_image_classifier = tf.keras.models.load_model(base_image_classifier_path)

# Load base meta-classifier
base_meta_classifier_path = meta_model_path + 'base_meta_model.sav'
base_meta_classifier = pickle.load(open(base_meta_classifier_path, 'rb'))

# Load base k_cluster
base_cluster_path = cluster_model_path + 'base_meta_k_means.sav'
base_meta_cluster = pickle.load(open(base_cluster_path, 'rb'))

multiple_detections_id = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r']
img_size = 528

hierarchy = {'base': ['Elephantidae', 'Felidae']}


def multiple_image_detections(index):
    images = []
    for possibility in multiple_detections_id:
        name = str(index) + '_' + possibility + '.jpg'
        file_path = image_path + name
        if os.path.exists(file_path):
            images.append(name)
        else:
            break
    return images


def preprocess_meta_data(df, k_means, taxon_target):
    # Remove non-essential columns
    df = df.drop(columns=['geoprivacy', 'taxon_geoprivacy', 'taxon_id', 'lat', 'long', 'time', 'observed_on_y',
                          'license', 'image_url', 'local_time_observed_at', 'positional_accuracy'])

    # Remove null species names
    df = df.dropna(subset=['taxon_species_name'])

    # Drop null positional accuracies
    df = df.dropna(subset=['public_positional_accuracy'])

    # Positional Accuracy Restriction
    df = df[df['public_positional_accuracy'] <= 40000]
    df = df.drop(columns=['public_positional_accuracy'])

    # Generate sub-species and drop scientific name
    df = df.apply(lambda x: sub_species_detection(x), axis=1)
    df = df.drop(columns=['scientific_name'])

    # Location Centroid Feature
    df['location_cluster'] = k_means.predict(df[['latitude', 'longitude']])

    # Terrestrial vs Land Feature
    df['land'] = 1

    # Elevation Logical Path, dependent on land
    df = df.apply(lambda x: elevation_clean(x), axis=1)
    df['elevation'] = df['elevation'].fillna(df.groupby('taxon_species_name')['elevation'].transform('mean'))

    # Northern and Southern Hemisphere OHE
    df['hemisphere'] = (df['latitude'] >= 0).astype(int)
    df = df.drop(columns=['latitude', 'longitude'])

    # Datetime Transformation
    df['observed_on'] = pd.to_datetime(df['observed_on'],
                                       format="%Y-%m-%d %H:%M:%S%z",
                                       utc=True)

    # Month Feature
    df['month'] = df['observed_on'].dt.month

    # Hour Feature
    df['hour'] = df.apply(lambda x: x['observed_on'].astimezone(pytz.timezone(x['time_zone'])).hour, axis=1)

    # Day/Night Feature
    df = day_night_calculation(df)

    # Season Feature
    df = df.apply(lambda x: season_calc(x), axis=1)
    df = ohe_season(df)

    # Drop observed on column as date & time transformations are complete
    df = df.drop(columns=['observed_on', 'time_zone'])

    # Retrieve labels
    taxon_y = df[taxon_target]

    # Sub-specie contains null values, if selected as target taxon. Remove
    if taxon_y.isnull().any():
        df = df.dropna(subset=[taxon_target])
    y = df[taxon_target]
    X = df.drop(columns=['taxon_kingdom_name', 'taxon_phylum_name',
                         'taxon_class_name', 'taxon_order_name', 'taxon_family_name',
                         'taxon_genus_name', 'taxon_species_name', 'sub_species', 'common_name'])
    return X, y




if __name__ == "__main__":
    data = pd.read_csv(data_path, index_col=0)

    X, y = preprocess_meta_data(data, base_meta_cluster, 'taxon_family_name')

    for index, obs in X.iterrows():
        current_taxon = 'base'
        obs = obs.to_numpy()
        print(index)
        meta_prediction = base_meta_classifier.predict_proba(obs.reshape(1, -1))
        print(meta_prediction)

        images = multiple_image_detections(index)
        for i in images:
            img = tf.keras.utils.load_img(image_path + i, target_size=(img_size, img_size))
            img = tf.keras.utils.img_to_array(img)
            input_arr = np.array([img])

            prediction = base_image_classifier.predict(input_arr)
            print(prediction)
            next_taxon = hierarchy[current_taxon][np.argmax(prediction)]
            print(next_taxon)