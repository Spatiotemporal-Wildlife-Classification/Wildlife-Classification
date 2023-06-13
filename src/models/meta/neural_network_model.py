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
    """This method specified the neural network modelling process

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
    """This method performs the neural network model training and hyperparameter tuning.

    Hyperparameter tuning aims to determine the optimal learning rate for each classification model.
    The learning rates tuned over include: [0.1, 0.01, 0.001, 0.0001]. Learning rate was selected due to the
    varying levels of abstractions within the taxonomic cascading structure.
    This process makes use of a best-model save policy based on the mean categorical accuracy (balanced accuracy) evaluation metric.

    Args:
        X (DataFrame): The input features to the decision tree
        y (Series): The categorical taxonomic labels of the corresponding observations to the features.
        classes (int): The number of unique classes for the model to classify
        model_name (str): The name of the model type being trained. In this case 'Neural network'.
        score_file (str): The filename of where the training data will be stored.
    """
    input_dimension = len(X.columns)  # determine the size of the input dimension
    epoch_num = 10  # Number of training epochs
    learning_rates = [0.1, 0.01, 0.001, 0.0001]  # Learning rates hyperparameter tuning is completed over
    fold_histories = []  # Training accuracy store
    best_accuracy = 0  # Best model accuracy initialization

    for rate in learning_rates:  # Iterate through learning rates
        kf_container = []  # Specify model store holders
        kf_val_accuracy = []
        kf_val_loss = []
        fold_ind = 0

        kf = KFold(n_splits=5)  # Initialize the 5-fold cross-validation

        for train_index, test_index in kf.split(X, y):
            X_train = X.iloc[train_index]  # Generate training dataset
            y_train = y[train_index]

            X_val = X.iloc[test_index]  # Generate test dataset
            y_val = y[test_index]

            y_cat = np.argmax(y_train, axis=1)  # Weight the training by presence of each class.
            unique_classes = np.unique(y_cat)
            weight_values = compute_class_weight(class_weight='balanced', classes=unique_classes, y=y_cat)
            weights = dict(zip(unique_classes, weight_values))

            model = make_model(input_dimension, classes)  # Create a new model

            opt = keras.optimizers.Adam(learning_rate=rate)  # Specify the models learning rate
            model.compile(optimizer=opt,
                          loss='categorical_crossentropy',
                          metrics=[keras.metrics.CategoricalAccuracy()])  # Compile the model.

            # Training
            history = model.fit(X_train.to_numpy(),
                                y_train,
                                epochs=epoch_num,
                                class_weight=weights,
                                verbose=0)  # Train model and record training history

            hist_df = pd.DataFrame(history.history)  # Save training categorical accuracy
            kf_container.append(hist_df['categorical_accuracy'].values.tolist())

            # Test
            results = model.evaluate(X_val, y_val, verbose=0)  # Generate model predictions for the test set
            kf_val_accuracy.append(results[1])  # Collect the categorical accuracy data
            kf_val_loss.append(results[0])  # Collect the loss

            fold_ind = fold_ind + 1  # Increase the current fold

        fold_histories.append(np.mean(kf_container, axis=0))  # Generate the training mean
        mean_accuracy = np.mean(kf_val_accuracy)  # Extract the mean accuracy from the model test
        mean_loss = np.mean(kf_val_loss)  # Extract the mean loss
        print(f"Mean accuracy with learning rate {rate} is {mean_accuracy} and loss is {mean_loss}")

        if best_accuracy < mean_accuracy:  # Best model save policy using test mean categorical accuracy and current best model.
            model.save(root_path + data_destination + model_name)
            best_accuracy = mean_accuracy
            print('Best model saved')

    write_training_accuracy(score_file, fold_histories, learning_rates)


def make_model(input_dimension: int, classes: int) -> keras.Sequential:
    """This method creates a new neural network model.

    This neural network has the following architecture:
    A variable input due to the varying number of input features.
    Two densely interconnected layers of 80 and 60 neurons each with RELU activation functions.
    Finally, a softmax output layer.

    Args:
        input_dimension (int): The number of input features to the neural network
        classes (int): The number of classes that should be classified in the output layer.

    Returns:
        (keras.Sequential): The model constructed in the specified architecture, with appropriate input and output dimensions.
    """
    model = keras.Sequential()
    model.add(Dense(input_dimension, input_shape=(input_dimension,), activation='relu'))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(classes, activation='softmax'))
    return model
