from tensorflow.keras.applications.efficientnet import EfficientNetB6
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.data import AUTOTUNE

import matplotlib.pyplot as plt

from src.structure.Config import root_dir

import numpy as np
import pandas as pd

# File path
root_path = root_dir()
img_path = root_path + '/data/wildlife_presence'
label_path = root_path + '/data/wildlife_presence.csv'

# Model Information
classes = 2
img_size = 528

# Dataset Information
batch_size = 32


# Modified Network allowing for transfer learning
def build_efficientnet():
    inputs = layers.Input(shape=(img_size, img_size, 3))  # Construct the expected image input
    model = EfficientNetB6(include_top=False,
                           input_tensor=inputs,
                           weights='imagenet')  # Initialize efficientnet model with imagenet weights
    model.trainable = False  # Freeze the pre-trained weights

    # Rebuild the top layers
    x = Sequential()
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(2048, activation='relu')(x)
    predictions = Dense(classes, activation='softmax')(x)

    # Initialize model
    model = Model(inputs=model.input, outputs=predictions, name='EfficientNet')
    optimizer = Adam(learning_rate=1e-2)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


def import_dataset(file_path: str):

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


def train_top_weights(train_ds, test_ds):
    model = build_efficientnet()
    epochs = 25
    hist = model.fit(train_ds, epochs=epochs, validation_data = test_ds, verbose=2)
    return model, hist


def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history['val_accuracy'])
    plt.title("Wildlife Presence Accuracy")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper_left")
    plt.show


if __name__ == "__main__":
    # Generate dataset and pre-tune
    train_ds, eval_ds = import_dataset(img_path)
    train_ds = train_ds.prefetch(AUTOTUNE)

    model, hist = train_top_weights(train_ds, eval_ds)
    plot_hist(hist)




