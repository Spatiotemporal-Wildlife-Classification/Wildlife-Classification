import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from keras.utils import image_dataset_from_directory
from sklearn.utils import compute_class_weight
from tensorflow.python.data import AUTOTUNE

import sys
import os

ilsvrc = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_l/feature_vector/2"

feature_model = ilsvrc
model_name = 'wildlife_presence_model_ilsvrc_3'

root_path = sys.path[1]
img_path = os.path.join(os.getcwd(), 'data', 'wildlife_presence')
save_path = os.path.join(os.getcwd(), 'models', model_name)

label_dict = {1: 'Present', 0: 'Absent'}


img_size = 480
classes = 2
batch_size = 32


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


def get_image_labels(ds):
    ohe_labels = []
    labels = []
    for x, y in ds:
        ohe_labels.extend(np.argmax(y.numpy().tolist(), axis=1))
    for label in ohe_labels:
        labels.append(label_dict[label])
    return labels


if __name__ == "__main__":
    # Print number of GPU's available
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Generate datset
    train_ds, eval_ds = import_dataset(img_path)

    # Create dataset weighting
    classes = train_ds.class_names
    weight_values = compute_class_weight(class_weight='balanced',
                                         classes=classes,
                                         y=get_image_labels(train_ds))
    print(weight_values)
    weights = dict(zip([0, 1], weight_values))

    # Prefetch the datset
    train_ds = train_ds.prefetch(AUTOTUNE)

    # Create feature extractor from pre-trained model
    feature_extractor_layer = hub.KerasLayer(
        feature_model,
        input_shape=(480, 480, 3),
        trainable=False
    )

    # Attach classification head
    num_classes = 2
    model = tf.keras.Sequential([
        feature_extractor_layer,
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    epochs = 20
    hist = model.fit(train_ds,
                     epochs=epochs,
                     validation_data=eval_ds,
                     verbose=2,
                     class_weight=weights)

    print(save_path)
    # Save model
    model.save(save_path)


