"""This file creates and trains the XGBoost metadata classification model.
    The XGBoost metadata classification model performs hyperparameter tuning over the depth of the XGBoost tree.
    The process makes use of 5-fold cross-validation to evaluate the models performance for each hyperparameter.
    A best-model save policy is enforced, using mean balanced accuracy as the evaluating metric.

    Attributes:
        root_path (str): The path to the project root.
        data_destination (str): The path to where the decision tree model and its training accuracy is saved.
"""

# Model
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score

# Project
import pipelines
from src.models.meta.decision_tree import write_scores_to_file
from src.structure import Config

# General
import numpy as np
import pandas as pd

root_path = Config.root_dir()
data_destination = pipelines.save_path


def xgboost_process(df: pd.DataFrame, taxon_target: str, model_name: str, score_file: str, validation_file: str):
    """This method specified the XGBoost modelling process

        Specifically this method, calls the required pipeline (XGBoost pipeline) to generate the features and labels required for training.
        Then, calls the training process to use the data.

        Args:
            df (DataFrame): The dataframe containing all data for each observation.
            taxon_target (str): The taxonomic target level, to extract the correct labels (taxon_family_name, taxon_genus_name, taxon_species_name, subspecies)
            model_name (str): The name of the model type being trained. In this case 'Decision tree'.
            score_file (str): The filename of where the training data will be stored. This will have the same location as where the model is saved.
            validation_file (str): The name of the file where the validation data will be stored. Also informs the name of the saved models.
    """
    X, y = pipelines.xgb_data(df, taxon_target, validation_file)  # Processing and formatting
    train_xgboost(X, y, model_name, score_file)  # Training and evaluation


def train_xgboost(X, y, model_name: str, score_file: str):
    """This method performs the XGBoost training and 5-fold cross-validation on the XGBoost tree depth as hyperparameter tuning to determine the optmial model.

    This process uses a best-model save policy based on the mean balanced accuracy evaluation metric.

    Args:
        X (DataFrame): The input features to the decision tree
        y (Series): The categorical taxonomic labels of the corresponding observations to the features.
        model_name (str): The name of the model type being trained. In this case 'XGBoost'.
        score_file (str): The filename of where the training data will be stored.
    """
    depth_limit = len(X.columns)  # Determine the maximum depth as the number of input features
    depth_range = range(1, depth_limit, 10)  # Specify the depth range with a increment of 10
    best_accuracy = 0  # Best balanced accuracy holder initialized
    scores = []  # Scores holder

    for depth in depth_range:  # Iterate through XGBoost tree depths as hyperparameter tuning
        kf = KFold(n_splits=5)  # Create the 5-fold cross-validation structure
        kf_score_container = []  # Cross validation scores holder
        fold_index = 0  # Keep track of current fold in 5-fold cross-validation

        for train_index, test_index in kf.split(X, y):
            X_train = X.iloc[train_index]  # Generate training dataset
            y_train = y[train_index]

            X_val = X.iloc[test_index]  # Generate test dataset
            y_val = y[test_index]

            y_cat = np.argmax(y_train, axis=1)  # Weight the training by presence of each class.
            weight_values = compute_sample_weight(class_weight='balanced', y=y_cat)

            model = xgb.XGBClassifier(random_state=0, n_jobs=5, max_depth=depth)  # Create the Xgboost Model

            model.fit(X_train, y_train, sample_weight=weight_values)  # Train the Xgboost model

            y_pred = np.argmax(model.predict(X_val), axis=1)  # Generate evaluation set predictions
            y_true = np.argmax(y_val, axis=1)  # Get true labels

            score = balanced_accuracy_score(y_true, y_pred)  # Calculate the balanced accuracy score
            print(f'{fold_index} fold accuracy is {score}')

            fold_index = fold_index + 1  # Increment current fold
            kf_score_container.append(score)  # Add score to the container

        mean_score = np.mean(kf_score_container)  # Determine the mean score of the 5-fold cross-validation
        print(f'Mean XGBoost score: {mean_score} for depth {depth}')
        scores.append(mean_score)  # Keep track of the mean score

        if best_accuracy < mean_score:  # Best model save policy comparing the best mean score from the 5-fold cross-validation
            file_name = root_path + data_destination + model_name
            model.save_model(file_name)
            best_accuracy = mean_score

    write_scores_to_file(scores, [*depth_range], score_file)
