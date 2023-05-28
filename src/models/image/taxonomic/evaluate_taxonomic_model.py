import os

import numpy as np
import pandas as pd
from keras.utils import image_dataset_from_directory
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.python.data import AUTOTUNE
import tensorflow as tf

img_size = 528
model_name = 'panthera_pardus_taxon_classifier'

test_path = os.path.join(os.getcwd(), 'data', 'taxon_test/felidae/panthera/panthera_pardus')
training_path = os.path.join(os.getcwd(), 'data', 'taxon/felidae/panthera/panthera_pardus/')
model_path = os.path.join(os.getcwd(), 'models/image/', model_name)
write_path = os.path.join(os.getcwd(),
                          'notebooks',
                          'taxon_image_classification_cache/image_classification_evaluation.csv')


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

    report_df.to_csv(write_path, mode='a', header=False)


if __name__ == "__main__":
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


    # Create confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Panthera Pardus Subspecies Level Classification')
    plt.show()
    resources_path = os.path.join(os.getcwd(), 'resources', 'panthera_pardus_subspecies_cm.jpg')
    plt.savefig(resources_path)

    # Create classification report
    report = classification_report(true_labels, predicted_labels)
    print(report)

    # Save results to file
    add_model_report(true_labels, predicted_labels, "Sub-species", classes)

