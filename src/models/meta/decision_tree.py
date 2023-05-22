import numpy as np

# Model
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import cross_val_score

# General
from src.structure import Config
import pipelines
import pandas as pd
import pickle


root_path = Config.root_dir()
data_destination = '/notebooks/model_comparison_cache/'
# data_destination = '/models/meta/'  # File save destination for use in ensemble


def write_scores_to_file(mean_scores: list, depth_range: list, filename: str):
    df = pd.DataFrame({'depth': depth_range, 'mean_scores': mean_scores})
    df.to_csv(root_path + data_destination + filename, index=False)


def decision_tree_process(df: pd.DataFrame, taxon_target: str, k_cluster, model_name: str, score_file: str, validation_file:str):
    X, y = pipelines.decision_tree_data(df, taxon_target, k_cluster, validation_file)
    train_decision_tree(X, y, model_name, score_file)


def train_decision_tree(X, y, model_name: str, score_file: str):
    depth_limit = len(X.columns)
    depth_range = range(1, depth_limit, 2)
    best_accuracy = 0
    scores = []

    for depth in depth_range:
        # Weight the training by presence of each class.
        classes = np.unique(y)
        weight_values = compute_class_weight(class_weight='balanced', classes=classes, y=y)
        weights = dict(zip(classes, weight_values))

        # Train the model
        clf = DecisionTreeClassifier(max_depth=depth, random_state=0, class_weight=weights)
        score = cross_val_score(estimator=clf,
                                X=X,
                                y=y,
                                cv=5,
                                n_jobs=-1,
                                scoring='balanced_accuracy')

        # Average the scores
        score_mean = np.mean(score)

        clf.fit(X.values, y)

        scores.append(score_mean)
        print(f"Depth {depth} out of {depth_limit}, generates {score_mean} accuracy")

        if best_accuracy < score_mean:
            # Save the best model
            filename = root_path + data_destination + model_name
            best_accuracy = score_mean
            pickle.dump(clf, open(filename, 'wb'))

    # Write mean scores/ loss to file
    write_scores_to_file(scores, [*depth_range], score_file)

