"""This file contains all methods needed to make, train, and evaluate the EfficientNet B6 model on the specified dataset.

    This file requires manual specification of the taxonomic parent node to model. Due to the massive memory and computational
    requirements of training a large CNN, only a single model can be trained at a time.

    Please note, the dataset must be structured within the taxonomic tree structure. Please review the `dataset_structure.py` file
    to see how this is accomplished..

    This training process is structured to be run within a Docker container in order to train on a single GPU unit.
    Please review the documentation or README how to run the training and validation processes.
    For easy access here is the command to run the model training:
    ```
    docker run --gpus all -u $(id -u):$(id -g) -v /path/to/project/root:/app/ -w /app -t model_train:latest
    ```

    Attributes:
        model_name (str): The saved name of the model. The file name must have the following format. taxonomic name + _taxon_classifier. Example: `lynx_lynx_taxon_classifier`
        img_path (str): The path to the taxonomic parent node within the `taxon` directory. Example" `felidae/lynx/lynx_lynx/`
        save_path (str): The path to where the model will be saved. In this case the `models/image/` directory
        img_size (int): The specified image size as input to the EfficientNet-B6 model (528)
        batch_size (int): The number of images within a single batch (32)
        epochs (int): The number of epochs in model training (25)
"""

# Modelling
from keras import Sequential, Model
from keras.applications import EfficientNetB6
from keras.layers import Dense, GlobalAveragePooling2D
from keras.utils import image_dataset_from_directory
from sklearn.utils import compute_class_weight
from tensorflow.python.data import AUTOTUNE
from tensorflow.keras import layers
import tensorflow as tf

# General
import os
import numpy as np
from matplotlib import pyplot as plt

# Data Information
model_name = ''
img_path = ''
save_path = ''

# Model details
img_size = 528
batch_size = 32
epochs = 25


def import_dataset(file_path: str):
    """This method imports the dataset from the proposed directory forming both a train and test set.

    This method uses the imaage_dataset_from_directory() method. For more information please visit:
    https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory

    This method, allows specification of the file path and automatically determines the labels based on the directory
    structure, hence the directory structure replicating the taxonomic tree of the dataset.
    Additionally, the class names are dispayed when this method is called.

    Args:
        file_path (str): The path from the `taxon/` (including) directory. Example: `taxon/felidae/lynx/lynx_lynx/`

    Returns:
        train_ds (tf.data.Dataset): The training dataset which will be used to train the model
        val_ds (tf.data.Dataset): The testing dataset used to test the trained model.
    """
    train_ds = image_dataset_from_directory(directory=file_path,
                                            validation_split=0.2,
                                            subset='training',
                                            seed=123,
                                            image_size=(img_size, img_size),
                                            batch_size=batch_size,
                                            labels='inferred',
                                            label_mode='categorical')
    val_ds = image_dataset_from_directory(directory=file_path,
                                          validation_split=0.2,
                                          subset='validation',
                                          seed=123,
                                          image_size=(img_size, img_size),
                                          batch_size=batch_size,
                                          labels='inferred',
                                          label_mode='categorical')
    print(train_ds.class_names)
    return train_ds, val_ds


def construct_model(classes: int):
    """Method constructs an EfficientNet-B6 model to fit the specified number of classes.

    The training makes use of transfer learning, so the EfficientNet-B6 model is created with ImageNet weights.
    The top softmax layer is removed from the original model and replaced with a Global Average Pooling 2D Layer, followed
    by a densely connected softmax layer classifying the specified number of classes.

    Args:
        classes (int): Integer specifying the number of classes to be classified. Instructs the size of the softmax output layer.

    Returns:
        (Model): The complete model ready to be trained.
    """
    inputs = layers.Input(shape=(img_size, img_size, 3))  # Construct the expected image input
    model = EfficientNetB6(include_top=False,
                           input_tensor=inputs,
                           weights='imagenet',
                           drop_connect_rate=0.2)  # Initialize efficientnet model with imagenet weights

    model.trainable = False  # Freeze the pre-trained weights

    # Rebuild the top layers
    x = Sequential()
    x = model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(classes, activation='softmax')(x)

    model = Model(inputs=model.input, outputs=predictions, name='EfficientNet_Taxon_Classifier')  # Construct the entire model

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Initialize optimizer
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])  # Compile the model for use

    return model


def get_image_labels(ds: tf.data.Dataset, classes: list):
    """Method generates class names from the dataset. This helps test the model

    Args:
        ds (tf.data.Dataset): Either the train or test dataset which labels must be generated for.
        classes (list): A list of the class labels (alphabetically ordered).

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


def train_model_top_weights(model: Model, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset):
    """Perform CNN training on the unfrozen top weights of the model.

    The dataset is weighted to achieve a balanced impact of each class on the model training.
    This is used to combat the long-tail distribution of the dataset.
    A best model save policy is created so only the best model from the training epochs is saved.

    Args:
        model (Model): The crated and prepared EfficientNet-B6 model with all but the top layers frozen, for training on the provided dataset.
        train_ds (tf.data.Dataset): The image training dataset
        val_ds (tf.data.Dataset): The image test dataset

    Returns:
        model (Model): The trained model
        hist (dict): The history of the model training process.
    """
    classes = train_ds.class_names   # Create dataset weighting
    weight_values = compute_class_weight(class_weight='balanced',
                                         classes=classes,
                                         y=get_image_labels(train_ds, classes))
    class_labels = list(range(0, len(classes)))
    weights = dict(zip(class_labels, weight_values))

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_path,
                                                     save_weights_only=False,
                                                     monitor='val_accuracy',
                                                     save_best_only=True,
                                                     verbose=1)  # Create the best-model save policy through the use of a callback

    train_ds = train_ds.prefetch(AUTOTUNE)  # Prefetch both the train and test dataset to speed up training and evaluation
    val_ds = val_ds.prefetch(AUTOTUNE)

    steps = int(len(train_ds) / batch_size)  # Specify training specifics. The calculation of steps and validation steps speeds up training
    validation_steps = int(0.05 * len(train_ds))

    if steps == 0:  # Due to varying dataset sizes, the conditionals determine if steps or validation steps are under 1 and correct for it.
        steps = 1
    if validation_steps == 0:
        validation_steps = 1

    print("Steps per epoch: ", steps)
    print("Validation steps: ", validation_steps)
    hist = model.fit(train_ds,
                     epochs=epochs,
                     validation_data=val_ds,
                     verbose=2,
                     callbacks=[cp_callback],
                     validation_steps=validation_steps,
                     steps_per_epoch=steps,
                     class_weight=weights)  # Train the model and gather training history
    return model, hist


def plot_hist(hist, title):
    """This method plots the accuracy and the validation set accuracy over the number of epochs and saves the figure.

    Args:
        hist (dict): A dictionary containing the training accuracy and testing accuracies per epoch
    """
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history['val_accuracy'])
    plt.title(title)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"])
    plt.show()  # Show the model

    save_title = title.replace(" ", "_")  # Prepare the figure save name
    save_title = save_title.lower()

    resources_path = os.path.join(os.getcwd(), 'resources', save_title + '.jpg')  # Save the figure
    plt.savefig(resources_path)


def train_model(file_name: str, dataset_path: str):
    """
    This model specifies the entire training process, and simplifies the model naming and dataset specification procedure for training.

    Args:
        file_name (str): The file name must have the following format. taxonomic name + _taxon_classifier. Example: `lynx_lynx_taxon_classifier`
        dataset_path (str): The path to the taxonomic parent node within the `taxon` directory. Example" `felidae/lynx/lynx_lynx/`
    """
    setup_paths(file_name, dataset_path)  # Setup the model save and dataset paths

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))  # This will print out the number of GPU's available

    train_ds, val_ds = import_dataset(img_path)  # Import and generate dataset
    classes = len(train_ds.class_names)  # Number of classes

    model = construct_model(classes)  # Construct the model

    model, hist = train_model_top_weights(model, train_ds, val_ds)  # Train the model's top weights

    try:  # Attempt to plot and save visualization of training and test data
        plot_hist(hist, "Lynx Lynx Classification Training")
    except:
        print('Not enough training epochs to generate display')


def setup_paths(file_name: str, dataset_path: str):
    """This method creates the correct file save and dataset access paths

    This method directly modifies global path variables

    Args:
        file_name (str): The file name must have the following format. taxonomic name + _taxon_classifier. Example: `lynx_lynx_taxon_classifier`
        dataset_path (str): The path to the taxonomic parent node within the `taxon` directory. Example" `felidae/lynx/lynx_lynx/`
    """
    global model_name, img_path, save_path
    model_name = file_name
    img_path = os.path.join(os.getcwd(), 'data', 'taxon/' + dataset_path)
    save_path = os.path.join(os.getcwd(), 'models/image/', model_name)


if __name__ == "__main__":
    train_model(file_name='elephantidae_taxon_classifier',
                dataset_path='elephantidae/')
