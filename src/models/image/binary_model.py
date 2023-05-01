import numpy as np
from tensorflow.keras.applications.efficientnet import EfficientNetB6
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.data import AUTOTUNE
import tensorflow as tf
from tensorflow import keras

import csv
import sys
import os

model_name = 'wildlife_presence_model_01'

# File path
root_path = sys.path[1]
img_path = os.path.join(os.getcwd(), 'data', 'wildlife_presence')
save_path = os.path.join(os.getcwd(), 'models', model_name)


# Model basics
img_size = 528
batch_size = 32
classes = 2

label_dict = {1: 'Present', 0: 'Absent'}


# Modified Network allowing for transfer learning (Version 01)
def build_efficientnet_01():
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
    optimizer = Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


# Modified Network allowing for transfer learning (Model 02)
def build_efficientnet_02():
    inputs = layers.Input(shape=(img_size, img_size, 3))  # Construct the expected image input
    model = EfficientNetB6(include_top=False,
                           input_tensor=inputs,
                           weights='imagenet')  # Initialize efficientnet model with imagenet weights
    model.trainable = False  # Freeze the pre-trained weights

    # Rebuild the top layers
    x = Sequential()
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)

    top_dropout_rate = 0.5
    x = Dropout(top_dropout_rate, name='top_dropout')(x)
    predictions = Dense(classes, activation='softmax')(x)

    # Initialize model
    model = Model(inputs=model.input, outputs=predictions, name='EfficientNet')
    optimizer = Adam(learning_rate=1e-4)
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


def train_top_weights(train_ds, eval_ds, model_type: int = 1):
    if model_type == 1:
        model = build_efficientnet_01()
    if model_type == 2:
        model = build_efficientnet_02()

    train_ds = train_ds.prefetch(AUTOTUNE)

    epochs = 10
    hist = model.fit(train_ds,
                     epochs=epochs,
                     validation_data=eval_ds,
                     verbose=2)
    return model, hist


def ohe_to_label(preds):
    y_pred = np.argmax(preds, axis=1)
    predicted_labels = []
    for pred in y_pred:
        predicted_labels.append(label_dict[pred])
    return predicted_labels


def get_image_labels(ds):
    ohe_labels = []
    labels = []
    for x, y in ds:
        ohe_labels.extend(np.argmax(y.numpy().tolist(), axis=1))
    for label in ohe_labels:
        labels.append(label_dict[label])
    return labels


def unfreeze_model(model):
    for layer in model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = Adam(learning_rate=1e-5)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics='accuracy')
    return model


def train_unfrozen_model(model, train_ds, eval_ds):
    epochs = 5

    hist = model.fit(train_ds,
                     epochs=epochs,
                     validation_data=eval_ds,
                     verbose=2)
    return model, hist


def write_training_history(file_name: str, accuracy):
    data_path = os.path.join(os.getcwd(), 'models', file_name)

    with open(data_path, 'w', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONE, delimiter=',')
        writer.writerows(accuracy)


if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Generate dataset and pre-tune
    train_ds, eval_ds = import_dataset(img_path)

    # Train the top weights of the model
    model, hist = train_top_weights(train_ds, eval_ds, 2)

    # # Unfreeze model
    # model = unfreeze_model(model)
    #
    # # Train model in its entirety
    # model, hist = train_unfrozen_model(model, train_ds, eval_ds)

    # Write training history
    print(hist.history['accuracy'])
    write_training_history('top_weight_model_01_training.csv', [hist.history['accuracy']])

    # Save model
    model.save(save_path)
