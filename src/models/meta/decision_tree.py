"""This file creates and trains the decision tree metadata classification model.

    The decision tree metadata classification model performs hyperparameter tuning over the depth of the decision tree.
    The training process makes use of 5-fold cross validation to evaluate the models performance for each hyperparameter.
    A best-model save policy is enforced using the mean accuracy across the 5-folds.


    Attributes:
        root_path (str): The path to the project root.
        data_destination (str): The path to where the decision tree model and its training accuracy is saved.
"""

# Model
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import cross_val_score

# Project
from src.structure import Config
import pipelines

# General
import pandas as pd
import pickle
import numpy as np


root_path = Config.root_dir()
data_destination = pipelines.save_path


def write_scores_to_file(mean_scores: list, depth_range: list, filename: str):
    """This method writes the model mean training and evaluation scores to a csv file for visualization and records.

    Note, the data written includes the mean training balanced accuracy value calculated in the cross validation.

    Args:
        mean_scores (list): The list of mean accuracy scores for each model.
        depth_range (list):: The list containing the decision tree depths trained over. For each element, there is a corresponding mean accuracy in mean_scores.
        filename (str): The filename, where the training data will be saved.
    """
    df = pd.DataFrame({'depth': depth_range, 'mean_scores': mean_scores})
    df.to_csv(root_path + data_destination + filename, index=False)


def decision_tree_process(df: pd.DataFrame, taxon_target: str, model_name: str, score_file: str, validation_file: str):
    """This method specifies the decision tree training process.

    Specifically this method, calls the required pipeline (decision tree pipeline) to generate the features and labels required for training.
    Then, calls the training process to use the data.

    Args:
        df (DataFrame): The dataframe containing all data for each observation.
        taxon_target (str): The taxonomic target level, to extract the correct labels (taxon_family_name, taxon_genus_name, taxon_species_name, subspecies)
        model_name (str): The name of the model type being trained. In this case 'Decision tree'.
        score_file (str): The filename of where the training data will be stored. This will have the same location as where the model is saved.
        validation_file (str): The name of the file where the validation data will be stored. Also informs the name of the saved models.
    """
    X, y = pipelines.decision_tree_data(df, taxon_target, validation_file)  # Processing and formatting
    train_decision_tree(X, y, model_name, score_file)  # Training and evaluation


def train_decision_tree(X, y, model_name: str, score_file: str):
    """This method performs the decision tree training and 5-fold cross validation on the tree depth hyperparameter to determine the optimal model

    This process uses a best-model save policy based on the balanced accuracy evaluation metric.

    Args:
        X (DataFrame): The input features to the decision tree
        y (Series): The categorical taxonomic labels of the corresponding observations to the features.
        model_name (str): The name of the model type being trained. In this case 'Decision tree'.
        score_file (str): The filename of where the training data will be stored.
    """
    depth_limit = len(X.columns)  # Calculate the depth limit as the number of input features
    depth_range = range(1, depth_limit, 2)  # Generate the depth range using an interval of 2
    best_accuracy = 0  # Instantiate the best accuracy holder
    scores = []  # Scores holder

    for depth in depth_range:  # Iterate through decision tree depths
        classes = np.unique(y)  # Weight the training by presence of each class.
        weight_values = compute_class_weight(class_weight='balanced', classes=classes, y=y)
        weights = dict(zip(classes, weight_values))  # Zip the calculated weights to the classes to form a dict

        clf = DecisionTreeClassifier(max_depth=depth, random_state=0, class_weight=weights)  # Create the model
        score = cross_val_score(estimator=clf,
                                X=X,
                                y=y,
                                cv=5,
                                n_jobs=-1,
                                scoring='balanced_accuracy')  # Train and evaluate the model using 5-fold cross-validation

        score_mean = np.mean(score)  # Average the scores

        clf.fit(X.values, y)  # Retrain the model for saving purposes if it is the top-performer

        scores.append(score_mean)  # Save mean score
        print(f"Depth {depth} out of {depth_limit}, generates {score_mean} accuracy")

        if best_accuracy < score_mean:  # Best model save policy
            filename = root_path + data_destination + model_name
            best_accuracy = score_mean
            pickle.dump(clf, open(filename, 'wb'))  # Save the model

    write_scores_to_file(scores, [*depth_range], score_file)  # Write mean scores/ loss to file

