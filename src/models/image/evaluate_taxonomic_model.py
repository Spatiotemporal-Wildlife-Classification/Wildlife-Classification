"""This file performs the evaluation on the trained CNN model recording the classification report and balanced accuracy metrics within a csv file for further use.

    Attributes:
        img_size (int): The specified image size as input to the EfficientNet-B6 model (528)
        model_name (str): The name of the CNN model to evaluate against a validation set.
        test_path (str): The path to the validation set of images to use to validate the model
        model_path (str): The path to the location of the model. Always `models/image/`
        report_path (os.path): The path to the csv file collecting all model classification reports
        accuracy_path (os.path): The path to the csv file collecting all model balanced accuracy values.
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
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    report_df = pd.DataFrame(report).transpose()
    report_df['taxon_level'] = taxon_level
    report_df = report_df.head(len(classes))

    report_df.to_csv(report_path, mode='a', header=False)


def add_model_accuracy(y_true, y_pred, taxon_level, taxon_name):
    accuracy_df = pd.DataFrame()
    accuracy = balanced_accuracy_score(y_true, y_pred)
    accuracy_df['accuracy'] = [accuracy]
    accuracy_df['taxon_level'] = [taxon_level]
    accuracy_df['taxon_name'] = taxon_name

    accuracy_df.to_csv(accuracy_path, mode='a', header=False, index=False)


def single_model_evaluation(current_model, path, taxon_level, display=False):
    global model_name, test_path, training_path, model_path
    model_name = current_model
    test_path = os.path.join(os.getcwd(), 'data', 'taxon_test/' + path)
    training_path = os.path.join(os.getcwd(), 'data', 'taxon/' + path)
    model_path = os.path.join(os.getcwd(), 'models/image/', model_name)
    taxon_name = current_model[:-17]


    # Generate test dataset
    test_ds = generate_test_set()

    # Generate training set to get classes
    train_ds = generate_training_set()

    # Gather class names
    classes = train_ds.class_names
    print(classes)

    true_labels = get_image_labels(test_ds, classes)

    # Load the saved model
    model = tf.keras.models.load_model(model_path)

    # Optimize for GPU running
    test_ds = test_ds.prefetch(AUTOTUNE)

    # Generate predictions for test set
    preds = model.predict(test_ds)
    preds = np.argmax(preds, axis=1)
    predicted_labels = np.take(classes, preds)

    if display:
        # Create confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(taxon_name + taxon_level + ' Level Classification')
        plt.show()

        resources_path = os.path.join(os.getcwd(), 'resources', taxon_name.lower() + taxon_level.lower() + '_cm.jpg')
        plt.savefig(resources_path)

    # Create classification report
    report = classification_report(true_labels, predicted_labels)
    print(report)

    # Save results to file
    add_model_report(true_labels, predicted_labels, taxon_level, classes)
    add_model_accuracy(true_labels, predicted_labels, taxon_level, taxon_name)


if __name__ == "__main__":
    single_model_evaluation('lynx_lynx_taxon_classifier', 'felidae/lynx/lynx_lynx/', 'Subspecies', False)


