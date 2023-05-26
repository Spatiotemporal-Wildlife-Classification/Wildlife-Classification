# General
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.utils import compute_sample_weight
from sklearn.ensemble import AdaBoostClassifier

from src.models.meta.decision_tree import write_scores_to_file
# Project
from src.structure import Config
import pipelines

root_path = Config.root_dir()
data_destination = '/notebooks/model_comparison_cache_2/'


def adaboost_process(df: pd.DataFrame, taxon_target: str, k_cluster, model_name: str, score_file: str, validation_file:str):
    X, y = pipelines.decision_tree_data(df, taxon_target, k_cluster, validation_file)
    train_adaboost(X, y, model_name, score_file)


def train_adaboost(X, y, model_name: str, score_file: str):
    estimator_range = range(1, 1000, 100)
    best_accuracy = 0
    scores = []

    for estimator in estimator_range:
        # Generate the cross-validation split
        kf = KFold(n_splits=5)
        kf_score_container = []
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
            model = AdaBoostClassifier(n_estimators=estimator, random_state=0)

            model.fit(X_train, y_train, weight_values)

            # Generate evaluation set predictions and true labels
            y_pred = model.predict(X_val)

            score = balanced_accuracy_score(y_val, y_pred)
            print(f'{fold_index} fold accuracy is {score}')
            fold_index = fold_index + 1
            kf_score_container.append(score)


        # Average the scores
        mean_score = np.mean(kf_score_container)
        print(f'Mean Adaboost score for {estimator} estimators: {mean_score}')
        scores.append(mean_score)

        if best_accuracy < mean_score:
            file_name = root_path + data_destination + model_name
            pickle.dump(model, open(file_name, 'wb'))
            best_accuracy = mean_score

    # Write mean scores/ loss to file
    write_scores_to_file(scores, [*estimator_range], score_file)
