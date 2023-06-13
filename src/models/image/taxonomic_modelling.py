"""This file contains all methods needed to make, train, and evaluate the EfficientNet B6 model on the specified dataset.

    This file requires manual specification of the taxonomic parent node to model. Due to the massive memory and computational
    requirements of training a large CNN, only a single model can be trained at a time.

    Please note, the dataset must be structured within the taxonomic tree structure. Please review the `dataset_structure.py` file
    to see how this is accomplished.

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
model_name = 'lynx_lynx_taxon_classifier'
img_path = os.path.join(os.getcwd(), 'data', 'taxon/felidae/lynx/lynx_lynx/')
save_path = os.path.join(os.getcwd(), 'models/image/', model_name)
checkpoint_path = os.path.join(os.getcwd(), 'models', 'checkpoints/sub_species')

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
        (Keras.model): The complete model ready to be trained.
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


def get_image_labels(ds, classes):
    ohe_labels = []
    labels = []
    for x, y in ds:
        ohe_labels.extend(np.argmax(y.numpy().tolist(), axis=1))
    for label in ohe_labels:
        labels.append(classes[label])
    return labels


def train_model_top_weights(model, train_ds, val_ds):
    # Create dataset weighting
    classes = train_ds.class_names
    weight_values = compute_class_weight(class_weight='balanced',
                                         classes=classes,
                                         y=get_image_labels(train_ds, classes))
    class_labels = list(range(0, len(classes)))
    weights = dict(zip(class_labels, weight_values))

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_path,
                                                     save_weights_only=False,
                                                     monitor='val_accuracy',
                                                     save_best_only=True,
                                                     verbose=1)

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    steps = int(len(train_ds) / batch_size)
    validation_steps = int(0.05 * len(train_ds))

    if steps == 0:
        steps = 1
    if validation_steps == 0:
        validation_steps = 1
    print("Steps per epoch: ", steps)
    hist = model.fit(train_ds,
                     epochs=epochs,
                     validation_data=val_ds,
                     verbose=2,
                     callbacks=[cp_callback],
                     validation_steps=validation_steps,
                     steps_per_epoch=steps,
                     class_weight=weights)
    return model, hist


def plot_hist(hist, title):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history['val_accuracy'])
    plt.title(title)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"])
    plt.show()
    save_title = title.replace(" ", "_")
    save_title = save_title.lower()
    resources_path = os.path.join(os.getcwd(), 'resources', save_title + '.jpg')
    plt.savefig(resources_path)


if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Import and generate dataset
    train_ds, val_ds = import_dataset(img_path)
    classes = len(train_ds.class_names)

    # Construct the model
    model = construct_model(classes)

    # Train the model's top weights
    model, hist = train_model_top_weights(model, train_ds, val_ds)

    try:
        plot_hist(hist, "Lynx Lynx Classification Training")
    except:
        print('Not enough training epochs to generate display')

    # model.save(save_path)
