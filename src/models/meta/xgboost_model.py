import numpy as np
import xgboost as xgb
import pandas as pd

import pipelines
from src.models.meta.decision_tree import write_scores_to_file
from src.structure import Config

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import KFold
from sklearn.metrics import balanced_accuracy_score

root_path = Config.root_dir()
# data_destination = '/notebooks/model_comparison_cache_2/'
data_destination = '/models/meta/'


def xgboost_process(df: pd.DataFrame, taxon_target: str, model_name: str, score_file: str,
                    validation_file: str):
    X, y = pipelines.xgb_data(df, taxon_target, validation_file)
    train_xgboost(X, y, model_name, score_file)


def train_xgboost(X, y, model_name: str, score_file: str):
    depth_limit = len(X.columns)
    depth_range = range(1, depth_limit, 10)
    best_accuracy = 0
    scores = []

    for depth in depth_range:
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
            y_cat = np.argmax(y_train, axis=1)
            weight_values = compute_sample_weight(class_weight='balanced', y=y_cat)

            # Create the Xgboost Model
            model = xgb.XGBClassifier(random_state=0,
                                      n_jobs=5,
                                      max_depth=depth)

            # Train the Xgboost model
            model.fit(X_train,
                      y_train,
                      sample_weight=weight_values)

            # Generate evaluation set predictions and true labels
            y_pred = np.argmax(model.predict(X_val), axis=1)
            y_true = np.argmax(y_val, axis=1)

            # Calculate the score
            score = balanced_accuracy_score(y_true, y_pred)
            print(f'{fold_index} fold accuracy is {score}')
            fold_index = fold_index + 1
            kf_score_container.append(score)

        mean_score = np.mean(kf_score_container)
        print(f'Mean XGBoost score: {mean_score} for depth {depth}')
        scores.append(mean_score)

        if best_accuracy < mean_score:
            file_name = root_path + data_destination + model_name
            model.save_model(file_name)
            best_accuracy = mean_score

    # Write mean scores/ loss to file
    write_scores_to_file(scores, [*depth_range], score_file)
