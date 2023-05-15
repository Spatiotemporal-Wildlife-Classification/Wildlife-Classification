import pickle

import pandas as pd
import numpy as np
import tensorflow as tf
from src.structure.Config import root_dir

import os

data_path = root_dir() + '/data/processed/final_test_observations.csv'
image_path = root_dir() + '/data/final_images/'
model_path = root_dir() + '/models/'

base_image_classifier_path = model_path + 'family_taxon_classifier'
base_image_classifier = tf.keras.models.load_model(base_image_classifier_path)

multiple_detections_id = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r']
img_size = 528


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


if __name__ == "__main__":
    data = pd.read_csv(data_path, index_col=0)
    for index, obs in data.iterrows():
        print(index)
        images = multiple_image_detections(index)
        for i in images:
            img = tf.keras.utils.load_img(image_path + i, target_size=(img_size, img_size))
            img = tf.keras.utils.img_to_array(img)
            input_arr = np.array([img])

            prediction = base_image_classifier.predict(input_arr)
            print(prediction)