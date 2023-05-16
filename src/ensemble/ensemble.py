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

tf.autograph.set_verbosity(2)
tf.get_logger().setLevel('ERROR')

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

# taxonomic_levels = ['taxon_family_name', 'taxon_genus_name', 'taxon_species_name', 'sub_species']
taxonomic_levels = ['taxon_family_name', 'taxon_genus_name']
hierarchy = {'base':
                 {'Elephantidae': {'Elephas': '',
                                   'Loxodonta': ''},
                  'Felidae': {'Acinonyx': '',
                              'Caracal': '',
                              'Catopuma': '',
                              'Felus': '',
                              'Herpailurus': '',
                              'Leopardus': '',
                              'Leptailurus': '',
                              'Lynx': '',
                              'Neofelis': '',
                              'Otocolobus': '',
                              'Panthera': '',
                              'Pardofelis': '',
                              'Prionailurus': '',
                              'Puma': ''}}}

# Meta data prediction weighting by taxonomic level
taxon_weighting = {'taxon_family_name': 0.1,
                   'taxon_genus_name': 0.2,
                   'taxon_species_name': 0.5,
                   'sub_species': 0.9}

# Method to identify multiple wildlife instances detected within a single image
def multiple_image_detections(index):
    images = []
    for possibility in multiple_detections_id:
        name = str(index) + '_' + possibility + '.jpg'  # Generate a possible image name and path
        file_path = image_path + name
        if os.path.exists(file_path):  # If the file exists, add it to images
            images.append(name)
        else:  # If no file exists, exit the loop no further images will be found
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


def taxon_weighted_decision(meta_prediction, image_prediction, taxon_level):
    weighting = taxon_weighting[taxon_level]
    weighted_meta = meta_prediction * weighting
    combined = weighted_meta + image_prediction
    return combined


def avg_multi_image_predictions(images, model):
    output_shape = model.output_shape[1:]
    mean_predictions = [0] * output_shape[0]
    for i in images:
        img = tf.keras.utils.load_img(image_path + i, target_size=(img_size, img_size))
        img = tf.keras.utils.img_to_array(img)
        input_arr = np.array([img])

        prediction = model.predict(input_arr, verbose=0)
        mean_predictions = mean_predictions + prediction
    return mean_predictions / len(images)


def load_next_meta_model(decision):
    name = decision.replace(" ", "_")
    name = name.lower()
    name = meta_model_path + name + '_dt_model.sav'
    model = pickle.load(open(name, 'rb'))
    return model


def load_next_cluster_model(decision):
    name = decision.replace(" ", "_")
    name = name.lower()
    name = cluster_model_path + name + '_dt_k_means.sav'
    model = pickle.load(open(name, 'rb'))
    return model


def load_next_image_model(decision):
    name = decision.replace(" ", "_")
    name = name.lower()
    name = image_model_path + name + '_taxon_classifier'
    model = tf.keras.models.load_model(name)
    return model


if __name__ == "__main__":
    data = pd.read_csv(data_path, index_col=0)

    X, y = preprocess_meta_data(data, base_meta_cluster, 'taxon_family_name')
    meta_model = base_meta_classifier
    image_model = base_image_classifier
    cluster_model = base_meta_cluster

    for index, obs in X.iterrows():
        print('---Image index: ', index, ' ---')

        current_level = hierarchy['base']  # Generate base hierarchy level
        obs = obs.to_numpy()  # Convert meta-data sample to numpy array

        for level in taxonomic_levels:
            print('-> ', level)

            # Image prediction
            images = multiple_image_detections(index)
            mean_img_prediction = avg_multi_image_predictions(images, image_model)
            print('Mean image prediction: ', mean_img_prediction)

            # Meta prediction
            X, y = preprocess_meta_data(data, cluster_model, level)
            meta_prediction = meta_model.predict_proba(obs.reshape(1, -1))
            print('Meta image prediction: ', meta_prediction)

            # Decision
            joint_prediction = taxon_weighted_decision(meta_prediction, mean_img_prediction, level)
            print('Joint prediction: ', joint_prediction)
            labels = (list(current_level.keys()))
            label = (labels[np.argmax(joint_prediction)])
            print('Predicted label: ', label)

            # Update hierarchy level
            current_level = current_level[labels[np.argmax(joint_prediction)]]

            # Update models based on next model
            meta_model = load_next_meta_model(label)
            image_model = load_next_image_model(label)
            cluster_model = load_next_cluster_model(label)

    print('--------------------------')