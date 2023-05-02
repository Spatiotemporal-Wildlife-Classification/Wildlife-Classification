import sys
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.utils import image_dataset_from_directory
from tensorflow.data import AUTOTUNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from src.models.image.imagenet.binary_model import ohe_to_label, get_image_labels

root_path = sys.path[1]
model_path = root_path + '/models/'
data_path = root_path + '/data/test/'

# Model basics
img_size = 528
batch_size = 32
classes = 2

label_dict = {1: 'Present', 0: 'Absent'}


def load_model(model_name: str):
    model = tf.keras.models.load_model(model_path + model_name)
    return model


def generate_test_set():
  test_ds = image_dataset_from_directory(directory=data_path,
                                         seed=123,
                                         image_size=(img_size, img_size),
                                         labels='inferred',
                                         label_mode='categorical')
  print(test_ds.class_names)
  return test_ds


def plot_hist(hist, title):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history['val_accuracy'])
    plt.title(title)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"])
    plt.show()


if __name__ == "__main__":
    # Load trained model
    model = load_model('wildlife_presence_model_03')

    # Generate test set
    test_ds = generate_test_set()


    # Pre-fetch test set
    test_ds = test_ds.prefetch(AUTOTUNE)

    # Generate predictions for test set
    preds = model.predict(test_ds)
    predicted_labels = ohe_to_label(preds)
    true_labels = get_image_labels(test_ds)

    # Create confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.title('Wildlife Presence ILSVCR 3')
    plt.show()

    report = classification_report(true_labels, predicted_labels)
    print(report)