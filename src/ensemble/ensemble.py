#!/usr/bin/env python3

import pickle
import sys
import time

import pandas as pd
import numpy as np
import schedule
import tensorflow as tf
import xgboost as xgb
import pytz

from csv import DictWriter

from src.models.meta.pipelines import elevation_clean, day_night_calculation, season_calc, ohe_season, \
    sub_species_detection
from src.structure.Config import root_dir

import gc
import os

data_path = root_dir() + '/data/processed/final_test_observations.csv'
results_path = root_dir() + '/notebooks/ensemble_cache_2/'
# results_path = root_dir() + '/notebooks/ensemble_comparison_cache/'
image_path = root_dir() + '/data/final_images/'
model_path = root_dir() + '/models/'
image_model_path = model_path + 'image/'
meta_model_path = model_path + 'meta/'
cluster_model_path = model_path + 'meta/'

# Load base image classifier
base_image_classifier_path = image_model_path + 'family_taxon_classifier'

# Load base meta-classifier
base_meta_classifier_path = meta_model_path + 'base_xgb_model.json'

# Load base k_cluster
base_cluster_path = cluster_model_path + 'base_xgb_k_means.sav'

multiple_detections_id = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r']
img_size = 528

taxonomic_levels = ['taxon_family_name', 'taxon_genus_name', 'taxon_species_name', 'sub_species']
hierarchy = {'base':
                 {'Elephantidae': {'Elephas':
                                       {'Elephas maximus':
                                            {'Elephas maximus borneensis': '',
                                             'Elephas maximus indicus': '',
                                             'Elephas maximus maximus': '',
                                             'Elephas maximus sumatranus': ''}},
                                   'Loxodonta':
                                       {'Loxodonta africana': '',
                                        'Loxodonta cyclotis': ''}},
                  'Felidae': {'Acinonyx':
                                  {'Acinonyx jubatus':
                                       {'Acinonyx jubatus hecki': '',
                                        'Acinonyx jubatus jubatus': ''}},
                              'Caracal':
                                  {'Caracal aurata':
                                       {'Caracal aurata aurata': ''},
                                   'Caracal caracal': ''},
                              'Catopuma':
                                  {'Catopuma temminckii':
                                       {'Catopuma temminckii moormensis': ''}},
                              'Felis':
                                  {'Felis chaus': '',
                                   'Felis lybica':
                                       {'Felis lybica cafra': '',
                                        'Felis lybica lybica': '',
                                        'Felis lybica ornata': ''},
                                   'Felis margarita': '',
                                   'Felis nigripes': '',
                                   'Felis silvestris':
                                       {'Felis silvestris caucasica': '',
                                        'Felis silvestris silvestris': ''}},
                              'Herpailurus':
                                  {'Herpailurus yagouaroundi': ''},
                              'Leopardus':
                                  {'Leopardus braccatus': '',
                                   'Leopardus colocola': '',
                                   'Leopardus emiliae': '',
                                   'Leopardus garleppi': '',
                                   'Leopardus geoffroyi': '',
                                   'Leopardus guigna':
                                       {'Leopardus guigna guigna': '',
                                        'Leopardus guigna tigrillo': ''},
                                   'Leopardus guttulus': '',
                                   'Leopardus jacobita': '',
                                   'Leopardus pajeros': '',
                                   'Leopardus pardalis':
                                       {'Leopardus pardalis mitis': '',
                                        'Leopardus pardalis pardalis': ''},
                                   'Leopardus tigrinus': '',
                                   'Leopardus wiedii': ''},
                              'Leptailurus':
                                  {'Leptailurus serval':
                                       {'Leptailurus serval constantina': '',
                                        'Leptailurus serval lipostictus': '',
                                        'Leptailurus serval serval': ''}},
                              'Lynx':
                                  {'Lynx canadensis': '',
                                   'Lynx lynx':
                                       {'Lynx lynx carpathicus': '',
                                        'Lynx lynx dinniki': '',
                                        'Lynx lynx isabellinus': '',
                                        'Lynx lynx lynx': '',
                                        'Lynx lynx wrangeli': ''},
                                   'Lynx pardinus': '',
                                   'Lynx rufus':
                                       {'Lynx rufus escuinapae': '',
                                        'Lynx rufus fasciatus': '',
                                        'Lynx rufus rufus': ''}},
                              'Neofelis':
                                  {'Neofelis diardi':
                                       {'Neofelis diardi borneensis': ''},
                                   'Neofelis nebulosa': ''},
                              'Otocolobus':
                                  {'Otocolumbus manul':
                                       {'Otocolumbus manul nigripectus': ''}},
                              'Panthera':
                                  {'Panthera leo':
                                       {'Panthera leo leo': '',
                                        'Panthera leo melanochaita': ''},
                                   'Panthera onca': '',
                                   'Panthera pardus':
                                       {'Panthera pardus delacouri': '',
                                        'Panthera pardus fusca': '',
                                        'Panthera pardus kotiya': '',
                                        'Panthera pardus melas': '',
                                        'Panthera pardus orientalis': '',
                                        'Panthera pardus pardus': '',
                                        'Panthera pardus tulliana': ''},
                                   'Panthera tigris':
                                       {'Panthera tigris tigris': ''},
                                   'Panthera uncia': ''},
                              'Pardofelis':
                                  {'Pardofelis marmorata': ''},
                              'Prionailurus':
                                  {'Prionailurus bengalensis':
                                       {'Prionailurus bengalensis bengalensis': '',
                                        'Prionailurus bengalensis euptilurus': ''},
                                   'Prionailurus javanensis':
                                       {'Prionailurus javanensis javanensis': '',
                                        'Prionailurus javanensis sumatranus': ''},
                                   'Prionailurus planiceps': '',
                                   'Prionailurus rubiginosus': '',
                                   'Prionailurus viverrinus':
                                       {'Prionailurus viverrinus viverrinus': ''}},
                              'Puma':
                                  {'Puma concolor':
                                       {'Puma concolor concolor': '',
                                        'Puma concolor couguar': ''}}}}}

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

    # Positional Accuracy Restriction
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

    y = df[taxon_target]
    X = df.drop(columns=['taxon_kingdom_name', 'taxon_phylum_name',
                         'taxon_class_name', 'taxon_order_name', 'taxon_family_name',
                         'taxon_genus_name', 'taxon_species_name', 'sub_species', 'common_name'])

    return X, y


def taxon_weighted_decision(meta_prediction, image_prediction, taxon_level):
    meta_weighting = taxon_weighting[taxon_level]
    image_weighting = 1 - meta_weighting
    weighted_meta = meta_prediction * meta_weighting
    weighted_image = image_prediction * image_weighting
    combined = weighted_meta + weighted_image
    combined = combined / np.sum(combined)  # Ensure a valid probability distribution
    return combined


def avg_multi_image_predictions(images, model):
    output_shape = model.output_shape[1:]
    mean_predictions = [0] * output_shape[0]
    for i in images:
        img = tf.keras.utils.load_img(image_path + i, target_size=(img_size, img_size))
        img = tf.keras.utils.img_to_array(img)
        input_arr = np.array([img])
        del img

        prediction = model.predict(input_arr, verbose=0)
        mean_predictions = mean_predictions + prediction
    mean_predictions = mean_predictions / len(images)
    return mean_predictions / np.sum(mean_predictions)


def load_next_meta_model(decision):
    if decision == 'base':
        model = xgb.XGBClassifier()
        model.load_model(base_meta_classifier_path)
        return model
    name = decision.replace(" ", "_")
    name = name.lower()
    name = meta_model_path + name + '_xgb_model.json'
    model = xgb.XGBClassifier()
    model.load_model(name)
    return model


def load_next_cluster_model(decision):
    if decision == 'base':
        model = pickle.load(open(base_cluster_path, 'rb'))
        return model
    name = decision.replace(" ", "_")
    name = name.lower()
    name = cluster_model_path + name + '_xgb_k_means.sav'
    model = pickle.load(open(name, 'rb'))
    return model


def load_next_image_model(decision):
    if decision == 'base':
        model = tf.keras.models.load_model(base_image_classifier_path)
        return model
    name = decision.replace(" ", "_")
    name = name.lower()
    name = image_model_path + name + '_taxon_classifier'
    model = tf.keras.models.load_model(name)
    return model


def instantiate_save_file():
    headings = ['id', 'taxonomic_level', 'joint_prediction', 'image_prediction', 'meta_prediction', 'true_label']
    f = open(results_path + 'ensemble_results.csv', 'a')
    dictwriter = DictWriter(f, fieldnames=headings)
    return dictwriter, f


def predict(index, data):
    print('---Image index: ', index, ' ---')

    writer, f = instantiate_save_file()

    current_level = hierarchy['base']  # Generate base hierarchy level

    label = 'base'

    for level in taxonomic_levels:
        print('-> ', level)
        try:
            labels = (list(current_level.keys()))  # Prepare prediction labels
        except:
            print(f"No {level} to be predicted")
            break

        if len(labels) == 1:
            label = labels[0]
            print('Single possibility: ', label)
            current_level = current_level[label]  # Update the current level based on the label
            continue

        # Update models based on next model
        meta_model = load_next_meta_model(label)
        image_model = load_next_image_model(label)
        cluster_model = load_next_cluster_model(label)

        # Image prediction
        images = multiple_image_detections(index)
        mean_img_prediction = avg_multi_image_predictions(images, image_model)

        # Meta prediction
        X, y = preprocess_meta_data(data, cluster_model, level)  # Preprocess data for each level
        if X.isnull().values.any():
            print("Meta data contains null values ")
            break
        obs = X.loc[index]  # Access current row
        obs = obs.to_numpy()  # Convert meta-data sample to numpy array
        meta_prediction = meta_model.predict_proba(obs.reshape(1, -1))

        # Decision
        joint_prediction = taxon_weighted_decision(meta_prediction, mean_img_prediction, level)
        label = (labels[np.argmax(joint_prediction)])

        # Update hierarchy level
        current_level = current_level[labels[np.argmax(joint_prediction)]]

        # True label

        if pd.isnull(y[index]):
            print(f"No label provided at {level}")
            break

        true_label = y[index]

        # Display
        print('Mean image prediction: ', mean_img_prediction)
        print('Meta image prediction: ', meta_prediction)
        print('Joint prediction: ', joint_prediction)
        print('Predicted label: ', label)
        print('True label: ', true_label)

        results = {'id': index,
                   'taxonomic_level': level,
                   'joint_prediction': label,
                   'image_prediction': labels[np.argmax(mean_img_prediction)],
                   'meta_prediction': labels[np.argmax(meta_prediction)],
                   'true_label': true_label}
        writer.writerow(results)

        del meta_model
        del cluster_model
        del image_model
        gc.collect()
        tf.keras.backend.clear_session()

        if true_label != label:
            print('Classification Mismatch')
            break
    f.close()
    print('--------------------------')


def read_position():
    with open('position.csv', 'r') as f:
        data = f.read()
        return int(data)

def update_position(prev_position):
    with open('position.csv', 'w') as f:
        f.write(str(prev_position + 1))
        f.close()


def ensemble_iteration():
    observation_no = read_position()

    data = pd.read_csv(data_path, index_col=0)  # Read in the final test dataset
    data = data.dropna(subset=['latitude', 'longitude', 'taxon_species_name'])

    if observation_no > len(data):
        quit()

    data = data.iloc[observation_no: observation_no + 1, :]

    for index, obs in data.iterrows():
        predict(index, data)

    update_position(observation_no)
    os.execv(sys.executable, [sys.executable] + sys.argv)


if __name__ == "__main__":
    while True:
        ensemble_iteration()
