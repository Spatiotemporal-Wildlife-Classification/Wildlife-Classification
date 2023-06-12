"""This file creates and trains the AdaBoost metadata classification model.
    The AdaBoost classification model performs hyperparameter tuning over the number of
    estimators to be used within the ensemble method. The number of estimators experimented over
    is within the range of [1, 201] using an increment of 20.
    The process makes use of 5-fold cross-validation to evaluate the models performance for each hyperparameter.
    A best-model save policy is enforced, using mean balanced accuracy as the evaluating metric.

    Attributes:
        root_path (str): The path to the project root.
        data_destination (str): The path to where the decision tree model and its training accuracy is saved.
"""

# Model
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
from sklearn.utils import compute_sample_weight
from sklearn.ensemble import AdaBoostClassifier

# General
import pickle
import numpy as np
import pandas as pd

# Project
from src.structure import Config
import pipelines
from src.models.meta.decision_tree import write_scores_to_file


root_path = Config.root_dir()
data_destination = pipelines.save_path


def adaboost_process(df: pd.DataFrame, taxon_target: str, model_name: str, score_file: str, validation_file: str):
    """This method specified the XGBoost modelling process

        Specifically this method, calls the required pipeline (Decision tree pipeline) to generate the features and labels required for training.
        Then, calls the training process to use the data.

        Args:
            df (DataFrame): The dataframe containing all data for each observation.
            taxon_target (str): The taxonomic target level, to extract the correct labels (taxon_family_name, taxon_genus_name, taxon_species_name, subspecies)
            model_name (str): The name of the model type being trained. In this case 'Adaboost'.
            score_file (str): The filename of where the training data will be stored. This will have the same location as where the model is saved.
            validation_file (str): The name of the file where the validation data will be stored. Also informs the name of the saved models.
    """
    X, y = pipelines.decision_tree_data(df, taxon_target, validation_file)  # Processing and formatting
    train_adaboost(X, y, model_name, score_file)  # Training and evaluation


def train_adaboost(X, y, model_name: str, score_file: str):
    """This method performs the Adaboost training and hyperparameter tuning.

    Hyperparameter tuning aims to determine the optimal number of estimators to be used within the Adaboost model,
    for each classifier.

    Args:
        X (DataFrame): The input features to the decision tree
        y (Series): The categorical taxonomic labels of the corresponding observations to the features.
        model_name (str): The name of the model type being trained. In this case 'XGBoost'.
        score_file (str): The filename of where the training data will be stored.
    """
    estimator_range = range(1, 201, 20)  # The range of estimators to be used in hyperparameter tuning
    best_accuracy = 0  # Best balanced accuracy holder initialized
    scores = []  # Scores holder

    for estimator in estimator_range:  # Iterate through Adaboost estimators as hyperparameter tuning
        kf = KFold(n_splits=5)  # Create the 5-fold cross-validation structure
        kf_score_container = []  # Cross validation scores holder
        fold_index = 0  # Keep track of current fold in 5-fold cross-validation

        for train_index, test_index in kf.split(X, y):
            X_train = X.iloc[train_index]  # Generate training dataset
            y_train = y[train_index]

            X_val = X.iloc[test_index]  # Generate test dataset
            y_val = y[test_index]

            weight_values = compute_sample_weight(class_weight='balanced', y=y_train)  # Weight the training by presence of each class.

            model = AdaBoostClassifier(n_estimators=estimator, random_state=0)  # Create Adaboost model

            model.fit(X_train, y_train, weight_values)  # Train the Adaboost model

            y_pred = model.predict(X_val)  # Generate test set predictions

            score = balanced_accuracy_score(y_val, y_pred)  # Calculate the balanced accuracy score
            print(f'{fold_index} fold accuracy is {score}')

            fold_index = fold_index + 1  # Increment current fold
            kf_score_container.append(score)  # Add score to the container

        mean_score = np.mean(kf_score_container)  # Determine the mean score of the 5-fold cross-validation
        print(f'Mean Adaboost score for {estimator} estimators: {mean_score}')
        scores.append(mean_score)  # Keep track of the mean score

        if best_accuracy < mean_score:  # Best model save policy comparing the best mean score from the 5-fold cross-validation
            file_name = root_path + data_destination + model_name
            pickle.dump(model, open(file_name, 'wb'))
            best_accuracy = mean_score

    write_scores_to_file(scores, [*estimator_range], score_file)
