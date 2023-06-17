"""This file forms the novel ensemble classifier.

    The novel ensemble classifier is a combination of two cascading taxonomic classification trees, using
    the metadata and image classifiers respectively at each parent node.
    The two trees, form a combined result which best evaluates the optimal prediction based on the classifiers strengths
    and when mitigating their weaknesses.

    This classifier operates to classify the validation dataset.
    The results serve as a comparison against baseline traditional image flat-classification methodologies.

    Please note, the hierarchy is hard coded due to the difficulty in getting the metadata and image classification models
    containing the same number of predicted children. This was hardcoded to achieve a result used to determine if this avenue of
    classification was worth pursuing. It is, so this method will need to be refined to be scalable, efficient, and distributed to
    be of use as a real-time classifier.
    For now, this script operates as a Proof of concept on the database.
    Adjust the hardcoded hierarchy if classifying wildlife on a different dataset.

    Attributes:
        data_path (str): The path to where the `validate.csv` dataset it. This is located in `data/obs_and_meta/processed/validation/`
        results_path (str): The path to where the results are stored. The results are stored within `notebooks/ensemble_model/ensemble_cache/` for easy visualizaiton in the Notebook.
        image_path (str): The path to the directory containing the validation images. The validation images and the data path sets are linked by observation id.
        model_path (str): The path to the base directory containing image and metadata classification models.
        image_model_path (str): The specific directory containing all image models (using `model_path` as the base path)
        meta_model_path (str): The specific directory containing all metadata models (using `model_path` as the base path)
        cluster_model_path (str): The specific directory containing all K-means model used to encode the observation locations. (using `model_path` as a base path)
        base_image_classifier_path (str): The path to the base image classifier. (The root image classifier classifying Felidae and Elephantidae as child classes)
        base_meta_classifier_path  (str): The path to the base metadata classifier. (The root metadata classifier classifying Felidae and Elephantidae as child classes)
        base_cluster_path (str): The path to the base K-means cluster model. (The models encoding the Felidae and Elephantidae possitions at the Family taxon level)
        multiple_detections_id (list): The list of possible image suffixes due to multiple sub-images per observation.
        img_size (int): The size of the input images to the image classifier (528, 528, 3)
        taxonomic_levels (list): The list of taxonomic levels at which classification occurs in the dataset from family to subspecies in order.
        hierarchy (dict): The taxonomic breakdown of the dataset.
        taxon_weighting (dict): The weighting of the metadata model predictions. The inverse presents the image classification predictions. These values are presented after observing the metadata and image classification taxonomic level experiment results.
"""
import keras
# Modelling
import tensorflow as tf
import xgboost as xgb
from sklearn.cluster import KMeans

# Project
from src.models.meta.pipelines import elevation_clean, day_night_calculation, season_calc, ohe_season, \
    sub_species_detection
from src.structure.Config import root_dir

# General
import gc
import os
import pytz
import pandas as pd
import numpy as np
import pickle
import sys
from csv import DictWriter

# Data paths
data_path = root_dir() + '/data/obs_and_meta/processed/validation.csv'
results_path = root_dir() + '/notebooks/ensemble_model/ensemble_cache/'
image_path = root_dir() + '/data/images/validation/'

# Model paths
model_path = root_dir() + '/models/'
image_model_path = model_path + 'image/'
meta_model_path = model_path + 'meta/'
cluster_model_path = model_path + 'k_clusters/'

# Models
base_image_classifier_path = image_model_path + 'family_taxon_classifier'  # Root image classifier path
base_meta_classifier_path = meta_model_path + 'base_xgb_model.json'  # Root metadata classifier path
base_cluster_path = cluster_model_path + 'base_xgb_k_means.sav'  # Root location encoding K-means model.

# Image details
multiple_detections_id = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r']  # Sub-image suffixes
img_size = 528

# Dataset taxonomy breakdown into nested dictionaries.
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

# Metadata prediction weighting by taxonomic level
taxon_weighting = {'taxon_family_name': 0.1,
                   'taxon_genus_name': 0.2,
                   'taxon_species_name': 0.5,
                   'sub_species': 0.9}


# Method to identify multiple wildlife instances detected within a single image
def multiple_image_detections(index):
    """This method gathers all sub-images per a single observation

    Due to the image pre-processing multiple sub-images can occur. Each sub-image centers and focuses on a identified
    wildlife individual. This method accumulated all file names based on the observation id.
    Sub-images are structured in the following format: `<id>_<alphabetical suffix>.jpg`

    Args:
        index (int): This is the unique id value of each observation.
    Returns:
        (list): A list of file names leading to sub-images of the specified observation id (index)
    """
    images = []
    for possibility in multiple_detections_id:  # Loop through the possible suffixes
        name = str(index) + '_' + possibility + '.jpg'  # Generate a possible image name and path
        file_path = image_path + name
        if os.path.exists(file_path):  # If the file exists, add it to images
            images.append(name)
        else:  # If no file exists, exit the loop no further images will be found
            break
    return images


def preprocess_meta_data(df: pd.DataFrame, k_means: KMeans, taxon_target: str):
    """This method processes and formats the metadata for model prediction, based on the taxon target the
    location encoding K-means model.

    This processing pipeline is essential for each dataset, as it processes and prepares the data based on the taxonomic level
    and the models pre-trained to suit the taxon level.
    This method looks similar to the data pipelines, but is modified due to the validation dataset already being processed.
    This pipeline formats the data into the required form for use.

    Args:
        df (DataFrame): The dataframe to be processed into features and labels.
        k_means (KMeans): The pre-trained location encoding model. This is trained based on the appropriate taxon parent node and taxon target, to be most effective within this dataset.
        taxon_target (str): The taxonomic target level, to extract the correct labels (taxon_family_name, taxon_genus_name, taxon_species_name, subspecies)

    Returns:
        X (DataFrame): The dataframe of features to be used by the metadata models
        y (Series): The labels of each observation, extracted at the taxonomic target level.
    """
    df = df.drop(columns=['geoprivacy', 'taxon_geoprivacy', 'taxon_id', 'lat', 'long', 'time', 'observed_on_y',
                          'license', 'image_url', 'local_time_observed_at', 'positional_accuracy'])  # Remove non-essential columns

    df = df[df['public_positional_accuracy'] <= 40000]  # Positional Accuracy Restriction
    df = df.drop(columns=['public_positional_accuracy'])

    # Generate subspecies and drop scientific name
    df = df.apply(lambda x: sub_species_detection(x), axis=1)
    df = df.drop(columns=['scientific_name'])

    df['location_cluster'] = k_means.predict(df[['latitude', 'longitude']])  # Location encoding using K-means

    df['land'] = 1  # All observations from dataset are terrestrial. For unknown datasets use the `land_mask()` method to automate the feature value

    df = df.apply(lambda x: elevation_clean(x), axis=1)   # Clean elevation values. In aquatic observations, the max elevation is sea level 0m
    df['elevation'] = df['elevation'].fillna(df.groupby('taxon_species_name')['elevation'].transform('mean'))

    df['hemisphere'] = (df['latitude'] >= 0).astype(int)  # Northern and Southern Hemisphere OHE
    df = df.drop(columns=['latitude', 'longitude'])

    df['observed_on'] = pd.to_datetime(df['observed_on'], format="%Y-%m-%d %H:%M:%S%z", utc=True)  # Datetime transform into datetime object
    df['month'] = df['observed_on'].dt.month  # Month Feature
    df['hour'] = df.apply(lambda x: x['observed_on'].astimezone(pytz.timezone(x['time_zone'])).hour, axis=1)  # Hour Feature (local time)
    df = day_night_calculation(df)  # Day/Night Feature
    df = df.apply(lambda x: season_calc(x), axis=1)  # Season feature into categorical values
    df = ohe_season(df)  # One-hot-encode the categorical season values

    df = df.drop(columns=['observed_on', 'time_zone'])  # Drop observed on column as date & time transformations are complete

    y = df[taxon_target]
    X = df.drop(columns=['taxon_kingdom_name', 'taxon_phylum_name',
                         'taxon_class_name', 'taxon_order_name', 'taxon_family_name',
                         'taxon_genus_name', 'taxon_species_name', 'sub_species', 'common_name'])
    return X, y


def taxon_weighted_decision(meta_prediction, image_prediction, taxon_level):
    """This method weights the metadata and image predictions based on the determined taxonomic weighting to produce
    a single softmax output to correctly predict the wildlife taxon

    The output is constrained to a valid probability distribution to match a softmax output.

    Args:
        meta_prediction (list): A list of metadata class probabilities. Note, the classes must match the image prediction
        image_prediction (list): A list of image class probabilities. Note, the classes must match the meta prediction.
        taxon_level (str): Specification of the taxon level, to specify the component weighting. (taxon_family_name, taxon_genus_name, taxon_species_name, sub_species)

    Returns:
        (list): A constrained softmax output constructed from the weighted influence of the individual metadata and image prediction components.
    """
    meta_weighting = taxon_weighting[taxon_level]  # Metadata prediction weight
    image_weighting = 1 - meta_weighting  # Image prediction weight is the inverse

    weighted_meta = meta_prediction * meta_weighting  # Apply weightings
    weighted_image = image_prediction * image_weighting

    combined = weighted_meta + weighted_image  # Combined and constrain weightings
    combined = combined / np.sum(combined)  # Ensure a valid probability distribution
    return combined


def avg_multi_image_predictions(images, model):
    """This model averages the predictions of sub-images to produce a single prediction per observation.

    This method combines the sub-image wildlife predictions together, averages them, and constrains them to a valid
    probability distribution to represent a softmax output, in order to provide a single prediction per
    observation (original image).

    Args:
         images (list): A list of sub-image paths of which predictions will be combined.
         model (keras.Sequential): The image classification model to classify the sub-images as the correct taxonomic level.

    Returns:
        (list): A summed, averaged, and constrained softmax output to provide a single image classification per observation.
    """
    output_shape = model.output_shape[1:]  # Determine the number of classes in the prediction
    mean_predictions = [0] * output_shape[0]  # Generate a container for the mean prediction

    for i in images:  # Loop through sub-images
        img = tf.keras.utils.load_img(image_path + i, target_size=(img_size, img_size))  # Format image
        img = tf.keras.utils.img_to_array(img)
        input_arr = np.array([img])
        del img  # Remove original image to save memory

        prediction = model.predict(input_arr, verbose=0)  # Classify image
        mean_predictions = mean_predictions + prediction  # Sum with previous predictions

    mean_predictions = mean_predictions / len(images)  # Average predictions
    return mean_predictions / np.sum(mean_predictions)  # Return constrained prediction


def load_next_meta_model(decision):
    """This method handles the loading of the correct metadata model, based on the provided decision

    Args:
        decision (str): The taxon label instructing the method which model to load. This is ordinarily the name of the taxonomic child. However "base" loads the root model.

    Returns:
        (xgboost): The pre-trained xgboost model classifying the children of the provided decision.
    """
    if decision == 'base':  # Root metadata model
        model = xgb.XGBClassifier()
        model.load_model(base_meta_classifier_path)
        return model

    name = decision.replace(" ", "_")  # Format taxon name to match model naming convention
    name = name.lower()
    name = meta_model_path + name + '_xgb_model.json'

    model = xgb.XGBClassifier()  # Load the model
    model.load_model(name)
    return model


def load_next_cluster_model(decision):
    """This method handles the loading of the correct K-means model, based on the decision

    Args:
        decision (str): The taxon label instructing the method which model to load. This is ordinarily the name of the taxonomic child. However "base" loads the root model.

    Returns:
        (KMeans): The pre-trained K-means model encoding the locations of the child nodes.
    """
    if decision == 'base':  # Root K-means encoding model
        model = pickle.load(open(base_cluster_path, 'rb'))
        return model

    name = decision.replace(" ", "_")  # Format taxon name to match model naming convention
    name = name.lower()
    name = cluster_model_path + name + '_xgb_k_means.sav'

    model = pickle.load(open(name, 'rb'))  # Load the model
    return model


def load_next_image_model(decision):
    """This method handles the loading of the correct image model, based on the provided decision

    Args:
        decision (str): The taxon label instructing the method which model to load. This is ordinarily the name of the taxonomic child. However "base" loads the root model.

    Returns:
        (keras.Sequential): The pre-trained EfficientNet-B6 model classifying the children of the provided decision.
    """
    if decision == 'base':  # Root image classification model
        model = tf.keras.models.load_model(base_image_classifier_path)
        return model

    name = decision.replace(" ", "_")  # Format taxon name to match model naming convention
    name = name.lower()
    name = image_model_path + name + '_taxon_classifier'

    model = tf.keras.models.load_model(name)  # Load the model
    return model


def instantiate_save_file():
    """This method instantiates the file saving and documenting the predictions of the ensemble model,
    its components, and the true labels.

    Returns:
        dictwriter (DictWriter): A dictionary writer object enabling dictionaries to be written to file f.
        f (file_handle): The file handle of the file to which the prediction data is being stored.
    """
    headings = ['id', 'taxonomic_level', 'joint_prediction', 'image_prediction', 'meta_prediction', 'true_label']
    f = open(results_path + 'ensemble_results.csv', 'a')
    dictwriter = DictWriter(f, fieldnames=headings)
    return dictwriter, f


def update_models(label: str):
    """This method serves to update all three models, based on the provided taxon label

    Args:
        label (str): The taxon label instructing the method which model to load. This is ordinarily the name of the taxonomic child. However "base" loads the root model.

    Returns:
        meta_model (KMeans): The metadata classification model for the label taxon parent node.
        image_model (keras.Sequential): The image classification model for the label taxon parent node.
        cluster_model (KMeans): The K-means encoding model for the children of the label taxon parent node.
    """
    meta_model = load_next_meta_model(label)
    image_model = load_next_image_model(label)
    cluster_model = load_next_cluster_model(label)
    return meta_model, image_model, cluster_model


def image_prediction(index: int, image_model: keras.Sequential):
    """This method handles the detection of sub-images and their mean image prediction.

    Args:
        index (int): The observation id providing the unique based identifier for sub-images of the original observation.
        image_model (keras.Sequential): The image classification model, for the current taxon parent node,

    Returns:
        (list): A summed, averaged, and constrained softmax output to provide a single image classification per observation.
    """
    images = multiple_image_detections(index)
    mean_img_prediction = avg_multi_image_predictions(images, image_model)
    return mean_img_prediction


def metadata_prediction(X: pd.DataFrame, index: int, meta_model: xgb):
    """This method handles the formatting of metadata and the model prediction

    Args:
        X (pd.DataFrame): The input features to the metadata model.
        index (int): The unique id of each observation.
        meta_model (xgb): The metadata model to predict the wildlife classes of the current taxon parent node.

    Returns:
        (list): The metadata model softmax prediction.
    """
    obs = X.loc[index]  # Access current row (current observation)
    obs = obs.to_numpy()  # Convert meta-data sample to numpy array
    meta_prediction = meta_model.predict_proba(obs.reshape(1, -1))  # Generate prediction
    return meta_prediction


def predict(index, data):
    """This method performs a cascading prediction for the specified observation.

    This method cascades from the family taxon to the subspecies taxon if labels are provided to that depth.
    The method documents the component, combined classifications, and true labels at each taxon level for analysis.
    The cascading process halts, when a miss-classification occurs.

    Args:
        index (int): The unique id of the current observation to be classified.
        data (pd.DataFrame): The dataframe containing all observation and metadata.
    """
    print('---Image index: ', index, ' ---')

    writer, f = instantiate_save_file()

    current_level = hierarchy['base']  # Generate base hierarchy level
    label = 'base'  # Current taxon label (parent taxon node)

    for level in taxonomic_levels:  # Decrease down the taxon levels predicting at each level, until a miss-classification or correctly classified
        print('-> ', level)
        try:
            labels = (list(current_level.keys()))  # Prepare prediction labels
        except:
            print(f"No {level} to be predicted")
            break

        if len(labels) == 1:  # There exists only a single possibility in the taxon tree, no need to predict
            label = labels[0]
            print('Single possibility: ', label)
            current_level = current_level[label]  # Update the current level based on the label
            continue

        meta_model, image_model, cluster_model = update_models(label)  # Update models based on next model

        mean_img_prediction = image_prediction(index, image_model)  # Image prediction

        X, y = preprocess_meta_data(data, cluster_model, level)  # Preprocess data for each taxon level
        if X.isnull().values.any():  # Emergency case where null values may interrupt process
            print("Meta data contains null values ")
            break
        meta_prediction = metadata_prediction(X, index, meta_model)  # Metadata prediction

        # Decision
        joint_prediction = taxon_weighted_decision(meta_prediction, mean_img_prediction, level)
        label = (labels[np.argmax(joint_prediction)])

        # Update hierarchy level
        current_level = current_level[labels[np.argmax(joint_prediction)]]

        if pd.isnull(y[index]):  # There is no taxon label provided at this level. This is to accomodate for subspecies where not every observation has a label
            print(f"No label provided at {level}")
            break

        true_label = y[index]  # True label

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
        writer.writerow(results)  # Save results to file

        del meta_model  # Clear memory so the next model can be loaded
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
