"""This file creates and trains the Random Forest metadata classification model.
    The Random Forest Classification model performs hyperparameter tuning on the tree depth used for each estimator in
    the ensemble method.
    The tree depth was from 1 to the number of input features. Input features are variable due to the cascading
    taxonomic structure.
    The process makes use of 5-fold cross-validation to evaluate the performance for each model.
    A best-model save policy is enforced, using the mean balanced accuracy as the evaluating metric

    Attributes:
        root_path (str): The path to the project root.
        data_destination (str): The path to where the decision tree model and its training accuracy is saved.
"""

# Model
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import compute_class_weight

# General
import numpy as np
import pickle
import pandas as pd

# Project
from src.structure import Config
from src.models.meta import pipelines
from src.models.meta.decision_tree import write_scores_to_file


root_path = Config.root_dir()
data_destination = pipelines.save_path


def random_forest_process(df: pd.DataFrame, taxon_target: str, model_name: str, score_file: str, validation_file: str):
    """This method specified the Random Forest modelling process

        Specifically this method, calls the required pipeline (Decision tree pipeline) to generate the features and labels required for training.
        Then, calls the training process to use the data.

        Args:
            df (DataFrame): The dataframe containing all data for each observation.
            taxon_target (str): The taxonomic target level, to extract the correct labels (taxon_family_name, taxon_genus_name, taxon_species_name, subspecies)
            model_name (str): The name of the model type being trained. In this case 'Random Forest'.
            score_file (str): The filename of where the training data will be stored. This will have the same location as where the model is saved.
            validation_file (str): The name of the file where the validation data will be stored. Also informs the name of the saved models.
    """
    X, y = pipelines.decision_tree_data(df, taxon_target, validation_file)   # Processing and formatting
    train_random_forest(X, y, model_name, score_file)  # Training and evaluation


def train_random_forest(X, y, model_name: str, score_file: str):
    """This method performs the Random Forest training and hyperparameter tuning.

    Hyperparameter tuning aims to determine the optimal tree depth for each estimator within the ensemble method, to
    produce optimal classification models.

    Args:
        X (DataFrame): The input features to the decision tree
        y (Series): The categorical taxonomic labels of the corresponding observations to the features.
        model_name (str): The name of the model type being trained. In this case 'XGBoost'.
        score_file (str): The filename of where the training data will be stored.
    """
    depth_limit = len(X.columns)  # Determine maximum depth as the number of input features
    depth_range = range(1, depth_limit, 2)  # Generate the tree depth range with an increment of 2
    best_accuracy = 0  # Best balanced accuracy holder initialized
    scores = []  # Scores holder

    for depth in depth_range:  # Iterate through the tree depth range
        classes = np.unique(y)  # Weight the training by presence of each class.
        weight_values = compute_class_weight(class_weight='balanced', classes=classes, y=y)
        weights = dict(zip(classes, weight_values))

        clf = RandomForestClassifier(max_depth=depth, random_state=0, n_jobs=5, class_weight=weights)  # Create Random Forest classifier
        score = cross_val_score(estimator=clf,
                                X=X,
                                y=y,
                                cv=5,
                                n_jobs=3,
                                scoring='balanced_accuracy')  # Train and evaluate using 5-fold cross-validation

        score_mean = np.mean(score)  # Average the balanced accuracy from the 5 folds
        scores.append(score_mean)  # Save the mean balanced accuracy
        print(f"Depth {depth} out of {depth_limit}, generates {score_mean} accuracy")

        clf.fit(X, y)  # Refit the model in case of saving

        if best_accuracy < score_mean:  # Best model save policy comparing the best mean score from the 5-fold cross-validation
            filename = root_path + data_destination + model_name
            pickle.dump(clf, open(filename, 'wb'))

    write_scores_to_file(scores, [*depth_range], score_file)

