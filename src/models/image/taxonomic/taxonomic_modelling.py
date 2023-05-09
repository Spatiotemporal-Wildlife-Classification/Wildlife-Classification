import os

import numpy as np
from keras import Sequential, Model
from keras.applications import EfficientNetB6
from tensorflow.keras.applications.vgg16 import VGG16
from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from keras.utils import image_dataset_from_directory
from matplotlib import pyplot as plt
from sklearn.utils import compute_class_weight
from tensorflow.python.data import AUTOTUNE

from tensorflow.keras import layers
import tensorflow as tf

model_name = 'elephantidae_taxon_classifier'
img_path = os.path.join(os.getcwd(), 'data', 'taxon/elephantidae/')
save_path = os.path.join(os.getcwd(), 'models', model_name)
checkpoint_path = os.path.join(os.getcwd(), 'models', 'checkpoints/genus')

img_size = 528
batch_size = 32
epochs = 20


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


def construct_model(classes: int):
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

    # Construct the model
    model = Model(inputs=model.input, outputs=predictions, name='EfficientNet_Taxon_Classifier')

    optimizer = tf.keras.optimizers.experimental.SGD(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

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

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    hist = model.fit(train_ds,
                     epochs=epochs,
                     validation_data=val_ds,
                     verbose=2,
                     callbacks=[cp_callback],
                     validation_steps=int(0.05 * len(train_ds)),
                     steps_per_epoch=25,
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

    plot_hist(hist, "Genus Elephantidae Classification Training")
    model.save(save_path)
