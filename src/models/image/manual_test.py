"""This file provides the capability to manually specify files to be evaluated by a model in order to manually check
    the outputs.

    This file is not to be used for extensive testing, but a quick method to ensure the model and prediction are
    working as expected, and to test the predictions of difficult images.

    Attributes:
        img_size (int): The specified image size as input to the EfficientNet-B6 model (528).
        img_path (str): The path to the image to be predicted using the CNN model. Please note the image should be in `taxon_validate` directory.
        model_path (str): The path to the model to be used to evaluate the image.
"""
# General
import tensorflow as tf
import numpy as np

# Project
from src.structure.Config import root_dir

img_size = 528

model_path = root_dir() + "/models/image/"   # Base model path. Specific model added in load_paths method
img_path = root_dir() + "/data/images/taxon_validate/"  # Base image path


def load_paths(model_name: str, image_path: str):
    """This method creates the correct paths to load the CNN model and access the image to be predicted

    Args:
        model_name (str): The name of the model to be loaded in order to predict the image. Example: `family_taxon_classifier`
        image_path (str): The path to the image to be predicted. This originates from the `taxon_validate` directory. Example: `elephantidae/loxodonta/4321448_a.jpg`
    """
    global img_path, model_path
    img_path = img_path + image_path
    model_path = model_path + model_name


def predict(model_name: str, image_path: str):
    """This method provides a simplified prediction process, where the model name and image path are required,
    and it will print the models prediction for viewing.

    In the provided example, the family taxon_train classification model is used to test if Elephantidae is predicted over Felidae.
    Due to the alphabetical ordering that Elephantidae comes before Felidae,
    we know it should be the first value will represent the Elephantidae class.
    Use the taxon_train directory to determine which class is predicted.
    This is a rough and quick method to confirm the model is working correctly.

    Args:
        model_name (str): The name of the model to be loaded in order to predict the image. Example: `family_taxon_classifier`
        image_path (str): The path to the image to be predicted. This originates from the `taxon_validate` directory. `elephantidae/loxodonta/4321448_a.jpg`
    """
    load_paths(model_name, image_path)

    model = tf.keras.models.load_model(model_path)  # Load the saved model
    image = tf.keras.utils.load_img(img_path, target_size=(528, 528))  # Load the image with size (528, 528)
    input_arr = tf.keras.utils.img_to_array(image)  # Transform the image into an array to be used as input for the model
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)  # Model predictions

    print(predictions)


if __name__ == "__main__":
    predict(model_name="family_taxon_classifier",
            image_path="elephantidae/loxodonta/9979057_a.jpg")
