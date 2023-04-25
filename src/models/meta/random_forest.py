import pickle

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import compute_class_weight

from src.models.meta import pipelines
from src.models.meta.decision_tree import write_scores_to_file

import numpy as np

from src.structure import Config

root_path = Config.root_dir()
data_destination = '/notebooks/model_comparison_cache/'


def random_forest_process(df: pd.DataFrame, taxon_target: str, k_cluster, model_name: str, score_file: str, validation_file:str):
    X, y = pipelines.decision_tree_data(df, taxon_target, k_cluster, validation_file)
    train_random_forest(X, y, model_name, score_file)


def train_random_forest(X, y, model_name: str, score_file: str):
    depth_limit = len(X.columns)
    depth_range = range(1, depth_limit, 2)
    best_accuracy = 0
    scores = []

    for depth in depth_range:
        # Weight the training by presence of each class.
        classes = np.unique(y)
        weight_values = compute_class_weight(class_weight='balanced', classes=classes, y=y)
        weights = dict(zip(classes, weight_values))

        clf = RandomForestClassifier(max_depth=depth, random_state=0, n_jobs=5, class_weight=weights)
        score = cross_val_score(estimator=clf,
                                X=X,
                                y=y,
                                cv=5,
                                n_jobs=3,
                                scoring='balanced_accuracy')

        # Average the scores
        score_mean = np.mean(score)
        scores.append(score_mean)
        print(f"Depth {depth} out of {depth_limit}, generates {score_mean} accuracy")

        clf.fit(X, y)
        if best_accuracy < score_mean:
            # Save the best model
            filename = root_path + data_destination + model_name
            pickle.dump(clf, open(filename, 'wb'))

    # Write mean scores/ loss to file
    write_scores_to_file(scores, [*depth_range], score_file)

