"""This file creates and trains the neural network metadata classification model.

    The neural network model performs learning rate hyperparameter tuning due to the variable levels of abstraction
    within the taxonomic tree.
    The training process makes use of 5-fold cross validation to evaluate the models performance for each
    hyperparameter, using balanced accuracy as the evaluating metric.
    A best-model save policy is enforced using the mean accuracy across the 5-folds.


    Attributes:
        root_path (str): The path to the project root.
        data_destination (str): The path to where the neural network model and its training accuracy is saved.
"""

# Modelling
from tensorflow import keras

from keras.layers import Dense
from sklearn.utils import compute_class_weight
from sklearn.model_selection import KFold

# Project
from src.structure import Config
import pipelines

# General
import numpy as np
import pandas as pd

root_path = Config.root_dir()
data_destination = pipelines.save_path


def write_training_accuracy(filename: str, fold_histories: dict, learning_rate: list):
    """This method writes the mean training and evaluation scores to a csv file for visualization and recording purposes.

    Note, the data written is the mean 5-fold categorical accuracy at each epoch of training.
    This is written for each learning rate used, serving as hyperparameter tuning.

    Args:
        filename (str): The filename, where the training data will be saved.
        fold_histories (str): The mean 5-fold categorical accuracy for each epoch of training for all models trained.
        learning_rate (str): The learning rate applied to the trained and evaluated model.
    """
    df = pd.DataFrame(fold_histories)
    df['learning_rate'] = learning_rate  # Add learning rate to the dataframe.
    df.to_csv(root_path + data_destination + filename, index=False)


def neural_network_process(df: pd.DataFrame, taxon_target: str, model_name: str, score_file: str, validation_file: str):
    """This method specified the neural network training process

    Specifically this method, calls the required pipeline (neural network pipeline) to generate the features and labels required for training.
    Then, calls the training process to use the data.

    Args:
        df (DataFrame): The dataframe containing all data for each observation.
        taxon_target (str): The taxonomic target level, to extract the correct labels (taxon_family_name, taxon_genus_name, taxon_species_name, subspecies)
        model_name (str): The name of the model type being trained. In this case 'Neural network'.
        score_file (str): The filename of where the training data will be stored. This will have the same location as where the model is saved.
        validation_file (str): The name of the file where the validation data will be stored. Also informs the name of the saved models.
    """
    X, y, classes = pipelines.neural_network_data(df, taxon_target, validation_file)  # Processing and formatting
    train_neural_network(X, y, classes, model_name, score_file)  # Training and evaluation


def train_neural_network(X, y, classes: int, model_name: str, score_file: str):
    input_dimension = len(X.columns)
    epoch_num = 10
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    fold_histories = []
    best_accuracy = 0

    for rate in learning_rates:
        kf_container = []
        kf_val_accuracy = []
        kf_val_loss = []
        fold_ind = 0
        kf = KFold(n_splits=5)

        for train_index, test_index in kf.split(X, y):
            # Generate test and validation training sets
            X_train = X.iloc[train_index]
            y_train = y[train_index]

            X_val = X.iloc[test_index]
            y_val = y[test_index]

            # Weight the training by presence of each class.
            y_cat = np.argmax(y_train, axis=1)
            unique_classes = np.unique(y_cat)
            weight_values = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_cat)
            weights = dict(zip(unique_classes, weight_values))

            # Create a new model
            model = make_model(input_dimension, classes)

            opt = keras.optimizers.Adam(learning_rate=rate)
            model.compile(optimizer=opt,
                          loss='categorical_crossentropy',
                          metrics=[keras.metrics.CategoricalAccuracy()])

            # Train model and record training history
            history = model.fit(X_train.to_numpy(), y_train, epochs=epoch_num, class_weight=weights, verbose=0)

            # Save training history
            hist_df = pd.DataFrame(history.history)
            kf_container.append(hist_df['categorical_accuracy'].values.tolist())

            # Validation
            results = model.evaluate(X_val, y_val, verbose=0)
            kf_val_accuracy.append(results[1])
            kf_val_loss.append(results[0])

            fold_ind = fold_ind + 1

        fold_histories.append(np.mean(kf_container, axis=0))
        mean_accuracy = np.mean(kf_val_accuracy)
        mean_loss = np.mean(kf_val_loss)
        print(f"Mean accuracy with learning rate {rate} is {mean_accuracy} and loss is {mean_loss}")

        if best_accuracy < mean_accuracy:
            model.save(root_path + data_destination + model_name)
            best_accuracy = mean_accuracy
            print('Best model saved')

    write_training_accuracy(score_file, fold_histories, learning_rates)


def make_model(input_dimension: int, classes: int) -> keras.Sequential:
    model = keras.Sequential()
    model.add(Dense(input_dimension, input_shape=(input_dimension,), activation='relu'))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(classes, activation='softmax'))
    return model