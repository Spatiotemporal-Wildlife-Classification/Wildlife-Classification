"""This file performs the evaluation on the trained CNN model recording the classification report and balanced
    accuracy metrics within a csv file for further use.

    To visualize and analyse the validation result metrics,
    please view `notebook/image_classification/image_classification_visualization`.
    The data collected is placed within the `notebook/image_classificaiton/taxon_image_classification_cache/`
    directory.

    This validation process is structured to be run within a Docker container in order to train on a single GPU unit.
    Please review the documentation or README how to run the training and validation processes.
    For easy access here is the command to run the model training:
    ```
    docker run --gpus all -u $(id -u):$(id -g) -v /path/to/project/root:/app/ -w /app -t ghcr.io/trav-d13/spatiotemporal_wildlife_classification/validate_image:latest
    ```
    -------------------------------------------------------------------------------------------------------------------

    Please note, when using this file to evaluate a flat-classification model, make use of the `global_mean_image_prediction()` method.
    This averages the predictions for sub-images into a single image.
    The following lines should also be included:

    `file_paths = test_ds.file_paths` before the dataset prefetching.

    `accumulated_score, file_true = global_mean_image_prediction(file_paths, preds, true_labels)` after the model predictions.

    These methods produce an averaged and uniform softmax prediction per image. Use the accumulated_score as a replacement
    within the `preds = np.argmax(accumulated_score, axis=1)` code.

    Please additionally changes the report and accuracy paths to access the `global_image_classification_results.csv`
    and `global_image_classification_accuracy.csv`

    Please additionally change the dataset to `species_train` and `species_validate` directories.

    Attributes:
        img_size (int): The specified image size as input to the EfficientNet-B6 model (528)
        model_name (str): The name of the CNN model to evaluate against a validation set.
        test_path (str): The path to the validation set of images to use to validate the model
        model_path (str): The path to the location of the model. Always `models/image/`
        report_path (os.path): The path to the csv file collecting all model classification reports. Please review notebook `notebook/image_classification/image_classification_visualization` for the data visualized.
        accuracy_path (os.path): The path to the csv file collecting all model balanced accuracy values. Please review notebook `notebook/image_classification/image_classification_visualization` for the data visualized.
"""

# General
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Model and evaluation
from keras.utils import image_dataset_from_directory
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, balanced_accuracy_score
from tensorflow.python.data import AUTOTUNE
import tensorflow as tf

# Model specifics
img_size = 528
model_name = ''
test_path = ''
training_path = ''
model_path = ''

# Data collection paths
report_path = os.path.join(os.getcwd(),
                           'notebooks',
                           'image_classification/taxon_image_classification_cache/image_classification_evaluation.csv')
accuracy_path = os.path.join(os.getcwd(),
                             'notebooks',
                             'image_classification/taxon_image_classification_cache/image_classification_accuracies.csv')


def generate_test_set():
    """This method generates the test/ validation dataset from the test_path. Note this is a completely separate dataset from the training process.
    The shuffle parameter is set to False in order to maintain the dataset order to align with the generated labels.

    Returns:
        (tf.data.Dataset): The validation dataset used to produced validation metrics for the produced model.
    """
    test_ds = image_dataset_from_directory(directory=test_path,
                                           seed=123,
                                           image_size=(img_size, img_size),
                                           labels='inferred',
                                           label_mode='categorical',
                                           shuffle=False,  # Don't shuffle as this influences the order of the labels
                                           interpolation='bicubic')
    return test_ds


def generate_training_set():
    """This method generates the original training dataset in order to gather all class labels trained over.

    This method looks at the training data to ensure that all labels are accounted for when determining prediction labels from the validation set.

    Returns:
        (tf.data.Dataset): The training dataset over which the model was trained.
    """
    test_ds = image_dataset_from_directory(directory=training_path,
                                           seed=123,
                                           image_size=(img_size, img_size),
                                           labels='inferred',
                                           label_mode='categorical',
                                           shuffle=False,
                                           interpolation='bicubic')
    return test_ds


def get_image_labels(ds, classes):
    """Method generates class names for the validation dataset, using the classes trained over.

    Due to difficulties importing the file methods within the Docker container, this is a duplicate method from taxonomic_modelling.py.

    Args:
        ds (tf.data.Dataset): The validation dataset
        classes (list): A list of the class labels (alphabetically ordered). This is sourced from the original training dataset

    Returns:
        (list): A list of labels in the provided dataset, in the same order as specified in the dataset.
    """
    ohe_labels = []  # Container to hold the ohe encoded version of the labels
    labels = []  # Container to hold the categorical labels

    for x, y in ds:
        ohe_labels.extend(np.argmax(y.numpy().tolist(), axis=1))  # Generate ohe encoded labels from dataset
    for label in ohe_labels:
        labels.append(classes[label])  # Transform ohe labels into categorical labels
    return labels


# Method to add classification report into a dataframe structure
def add_model_report(y_true, y_pred, taxon_level, classes):
    """This method adds the classification report to the report csv file.

    Note, only the rows describing each class are added to the report. The end 3 rows \
    (accuracy, macro_avg, weighted avg) are excluded from being written to the report file.
    If the validation set is missing labels over which it is validated, it may include the last 3 rows, which would
    required manual removal.

    Args:
        y_true (list): The list of True labels
        y_pred (list): The list of predicted labels'
        taxon_level (str): The taxonomic target level.
        classes (list): The list of classes (alphabetical order) over which the model was trained.
    """
    report = classification_report(y_true, y_pred, output_dict=True,
                                   zero_division=0)  # Generate the classification report

    report_df = pd.DataFrame(report).transpose()  # Format the report into a dataframe
    report_df['taxon_level'] = taxon_level  # Add additional columns
    report_df = report_df.head(
        len(classes))  # Specify the row cutoff to avoid including the last 3 rows (accuracy, macro avg, weighted avg)

    if os.path.exists(report_path):
        report_df.to_csv(report_path, mode='a', header=False)
    else:
        report_df.to_csv(report_path, mode='w', header=True)


def add_model_accuracy(y_true, y_pred, taxon_level, taxon_name):
    """This method adds the models balanced accuracy metric to the csv file containing all balanced accuracies for
    further visualization and analysis.

    Args:
        y_true (list): The list of True labels
        y_pred (list): The list of predicted labels'
        taxon_level (str): The taxonomic target level.
        taxon_name (str): The standardized name of the taxonomic parent node for which the classifier is built
    """
    accuracy_df = pd.DataFrame()  # Initialize dataframe
    accuracy = balanced_accuracy_score(y_true, y_pred)  # Generate balanced accuracy metric
    print("Balanced accuracy: ", accuracy)

    # Write additional columns
    accuracy_df['accuracy'] = [accuracy]
    accuracy_df['taxon_level'] = [taxon_level]
    accuracy_df['taxon_name'] = taxon_name

    if os.path.exists(accuracy_path):
        accuracy_df.to_csv(accuracy_path, mode='a', header=False, index=False)
    else:
        accuracy_df.to_csv(accuracy_path, mode='w', header=True, index=False)


def global_mean_image_prediction(image_paths: list, predicted_labels: list, true_labels: list):
    """This method is used within the Flat-classification models to aggregate, and normalize the sub-image predictions
    into a single prediction for each image

    Args:
        image_paths (list): A list of file paths for each image in the dataset. Read main file documentation for additional info. The code to generate the filenames is `file_paths = test_ds.file_paths`
        predicted_labels (list): The list of predicted labels for all sub-images
        true_labels (list): The list of true labels for all sub-images. They are in the same order.

    Returns:
        mean_predictions (list): The summed, averaged, and normalized predictions for a single image (not a sub-image).
        individual_file_label (list): The list of true labels for each image. Of the same size and order as the mean_predictions.
    """
    files_modified = [image_path[:-6] for image_path in
                      image_paths]  # Modify file paths to exclude the _a.png, _b.png, ...

    accumulation_store = dict()  # Keep track of each image sum predictions
    path_counter = dict()  # Keep track of the number of sub-images for each image
    individual_file_label = []  # Keep track of the single true labels for each file

    for image_path, predicted_label, true_label in zip(files_modified, predicted_labels, true_labels):
        if image_path in accumulation_store:  # Image file has been predicted before, add to the prediction sum and increase coutner
            accumulation_store[image_path] = accumulation_store[image_path] + predicted_label
            path_counter[image_path] = path_counter[image_path] + 1
        else:  # New image detected. Add to dictionary and place initial prediction. Increase counter
            accumulation_store[image_path] = predicted_label
            path_counter[image_path] = 1
            individual_file_label.append(true_label)  # Append true label

    mean_predictions = []  # Store of mean and normalized predictions (maintain softmax output)
    for key, value in accumulation_store.items():
        mean_prediction = value / path_counter[key]  # Create the mean prediction
        normalized_prediction = mean_prediction / np.sum(mean_prediction)  # normalize the results
        mean_predictions.append(normalized_prediction)

    return mean_predictions, individual_file_label  # Return the predictions for each image and the true labels (same size)


def set_paths(current_model, path):
    """This method sets the essential paths to model and data resources

    Args:
        current_model (str): The name of the model to validate. Must adhere to naming conventions of taxonomic modelling. Example: `elephantidae_taxon_classifier`
        path (str): The path to the validation directory. This is the path within `taxon_validate` to the correct directory. Example: `elephantidae/`
    """
    global model_name, test_path, training_path, model_path

    model_name = current_model
    test_path = os.path.join(os.getcwd(), 'data', 'images/taxon_validate/' + path)
    training_path = os.path.join(os.getcwd(), 'data', 'images/taxon_train/' + path)
    model_path = os.path.join(os.getcwd(), 'models/image/', model_name)  # For global models change to 'models/global`


def single_model_evaluation(current_model, path, taxon_level, display=False):
    """This method provides a simple means of validating a trained CNN image model, through
    simple specification of the model name and dataset path.


    Args:
        current_model (str): The name of the model to validate. Must adhere to naming conventions of taxonomic modelling. Example: `elephantidae_taxon_classifier`
        path (str): The path to the validation directory. This is the path within `taxon_validate` to the correct directory. Example: `elephantidae/`
        taxon_level (str): The taxonomic level at which classification takes place (Family, Genus, Species, Subspecies). This is the level of the taxonomic children being classified.
        display (bool): A boolean value indicating whether the Confusion matrix of the model validation should be created and saved.
    """
    set_paths(current_model, path)  # Set the essential paths
    taxon_name = current_model[:-17]

    test_ds = generate_test_set()  # Generate validation dataset

    train_ds = generate_training_set()  # Generate training set to get classes
    classes = train_ds.class_names  # Gather class names
    print(classes)  # Display class names

    true_labels = get_image_labels(test_ds, classes)  # Generate true labels from the validation dataset

    model = tf.keras.models.load_model(model_path)  # Load the saved model

    test_ds = test_ds.prefetch(AUTOTUNE)  # Optimize for GPU running

    preds = model.predict(test_ds)  # Generate predictions for validation set

    preds = np.argmax(preds, axis=1)  # Get index value of the prediction
    predicted_labels = np.take(classes, preds)  # Extract class name from the index value

    if display:  # Create confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(taxon_name + taxon_level + ' Level Classification')
        plt.show()

        resources_path = os.path.join(os.getcwd(), 'resources',
                                      taxon_name.lower() + taxon_level.lower() + '_cm.jpg')  # Save confusion matrix
        plt.savefig(resources_path)

    report = classification_report(true_labels, predicted_labels)  # Create classification report
    print(report)

    # Save results to file
    add_model_report(true_labels, predicted_labels, taxon_level, classes)  # Save report to file
    add_model_accuracy(true_labels, predicted_labels, taxon_level, taxon_name)  # Save balanced accuracy to file


if __name__ == "__main__":
    """Method to execute the model validation process. 
    """
    single_model_evaluation(current_model='elephantidae_taxon_classifier',
                            path='elephantidae',
                            taxon_level='Genus',
                            display=False)
