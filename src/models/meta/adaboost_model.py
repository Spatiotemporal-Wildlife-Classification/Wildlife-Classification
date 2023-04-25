# General
import pickle

import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold
from sklearn.utils import compute_sample_weight
from sklearn.ensemble import AdaBoostClassifier

# Project
from src.structure import Config
import pipelines

root_path = Config.root_dir()
data_destination = '/notebooks/model_comparison_cache/'


def adaboost_process(df: pd.DataFrame, taxon_target: str, k_cluster, model_name: str, score_file: str, validation_file:str):
    X, y = pipelines.decision_tree_data(df, taxon_target, k_cluster, validation_file)
    folds = 4
    kf = KFold(n_splits=folds)
    train_adaboost(X, y, kf, model_name, score_file)


def train_adaboost(X, y, kf, model_name: str, score_file: str):
    best_accuracy = 0
    fold_index = 0

    for train_index, test_index in kf.split(X, y):
        # Generate test and validation training sets
        X_train = X.iloc[train_index]
        y_train = y[train_index]

        X_val = X.iloc[test_index]
        y_val = y[test_index]

        # Weight the training by presence of each class.
        weight_values = compute_sample_weight(class_weight='balanced', y=y_train)

        # Create Adaboost model
        model = AdaBoostClassifier(random_state=0)

        # Train the AdaBoost model
        model.fit(X_train, y_train, weight_values)

        # Generate evaluation set predictions and true labels
        y_pred = model.predict(X_val)

        # Calculate the score
        score = balanced_accuracy_score(y_val, y_pred)
        print(f'{fold_index} fold accuracy is {score}')
        fold_index = fold_index + 1

        if best_accuracy < score:
            file_name = root_path + data_destination + model_name
            pickle.dump(model, open(file_name, 'wb'))
            best_accuracy = score

