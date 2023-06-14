"""This file performs the evaluation on the trained CNN model recording the classification report and balanced
    accuracy metrics within a csv file for further use.

    To visualize and analyse the validation result metrics,
    please view `notebook/image_classification/image_classification_visualization`.
    The data collected in s placed within the `notebook/image_classificaiton/taxon_image_classification_cache/`
     directory.

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
                          'taxon_image_classification_cache/image_classification_evaluation.csv')
accuracy_path = os.path.join(os.getcwd(),
                          'notebooks',
                          'taxon_image_classification_cache/image_classification_accuracies.csv')


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
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)  # Generate the classification report

    report_df = pd.DataFrame(report).transpose()  # Format the report into a dataframe
    report_df['taxon_level'] = taxon_level  # Add additional columns
    report_df = report_df.head(len(classes))  # Specify the row cutoff to avoid including the last 3 rows (accuracy, macro avg, weighted avg)

    report_df.to_csv(report_path, mode='a', header=False)


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

    # Write additional columns
    accuracy_df['accuracy'] = [accuracy]
    accuracy_df['taxon_level'] = [taxon_level]
    accuracy_df['taxon_name'] = taxon_name

    accuracy_df.to_csv(accuracy_path, mode='a', header=False, index=False)


def set_paths(current_model, path):
    """This method sets the essential paths to model and data resources

    Args:
        current_model (str): The name of the model to validate. Must adhere to naming conventions of taxonomic modelling. Example: `elephantidae_taxon_classifier`
        path (str): The path to the validation directory. This is the path within `taxon_test` to the correct directory. Example: `elephantidae/`
    """
    global model_name, test_path, training_path, model_path

    model_name = current_model
    test_path = os.path.join(os.getcwd(), 'data', 'taxon_test/' + path)
    training_path = os.path.join(os.getcwd(), 'data', 'taxon/' + path)
    model_path = os.path.join(os.getcwd(), 'models/image/', model_name)


def single_model_evaluation(current_model, path, taxon_level, display=False):
    """This method provides a simple means of validating a trained CNN image model, through
    simple specification of the model name and dataset path.


    Args:
        current_model (str): The name of the model to validate. Must adhere to naming conventions of taxonomic modelling. Example: `elephantidae_taxon_classifier`
        path (str): The path to the validation directory. This is the path within `taxon_test` to the correct directory. Example: `elephantidae/`
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

    preds = model.predict(test_ds) # Generate predictions for validation set
    preds = np.argmax(preds, axis=1)  # Get index value of the prediction
    predicted_labels = np.take(classes, preds)  # Extract class name from the index value

    if display:  # Create confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(taxon_name + taxon_level + ' Level Classification')
        plt.show()

        resources_path = os.path.join(os.getcwd(), 'resources', taxon_name.lower() + taxon_level.lower() + '_cm.jpg')  # Save confusion matrix
        plt.savefig(resources_path)

    report = classification_report(true_labels, predicted_labels)  # Create classification report
    print(report)

    # Save results to file
    add_model_report(true_labels, predicted_labels, taxon_level, classes)  # Save report to file
    add_model_accuracy(true_labels, predicted_labels, taxon_level, taxon_name)  # Save balanced accuracy to file


if __name__ == "__main__":
    """Method to execute the model validation process. 
    """
    single_model_evaluation('elephantidae_taxon_classifier', 'elephantidae/', 'Genus', False)


