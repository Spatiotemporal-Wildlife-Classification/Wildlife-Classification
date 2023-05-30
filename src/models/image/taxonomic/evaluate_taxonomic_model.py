import os
import sys

import numpy as np
import pandas as pd
from keras.utils import image_dataset_from_directory
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, balanced_accuracy_score, accuracy_score
from tensorflow.python.data import AUTOTUNE
import tensorflow as tf

img_size = 528
model_name = ''
test_path = ''
training_path = ''
model_path = ''

report_path = os.path.join(os.getcwd(),
                          'notebooks',
                          'taxon_image_classification_cache/global_image_classification_evaluation.csv')
accuracy_path = os.path.join(os.getcwd(),
                          'notebooks',
                          'taxon_image_classification_cache/global_image_classification_accuracies.csv')


def generate_test_set():
    test_ds = image_dataset_from_directory(directory=test_path,
                                           seed=123,
                                           image_size=(img_size, img_size),
                                           labels='inferred',
                                           label_mode='categorical',
                                           shuffle=False,
                                           interpolation='bicubic')
    return test_ds


def generate_training_set():
    test_ds = image_dataset_from_directory(directory=training_path,
                                           seed=123,
                                           image_size=(img_size, img_size),
                                           labels='inferred',
                                           label_mode='categorical',
                                           shuffle=False,
                                           interpolation='bicubic')
    return test_ds


def get_image_labels(ds, classes):
    ohe_labels = []
    labels = []
    for x, y in ds:
        ohe_labels.extend(np.argmax(y.numpy().tolist(), axis=1))
    for label in ohe_labels:
        labels.append(classes[label])
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
    accuracy = accuracy_score(y_true, y_pred)
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
    single_model_evaluation('global_taxon_classifier', '', 'Global', False)


