"""This file performs metadata model training for all proposed models, at all taxonomic levels.

    The metadata model training is automated to train the proposed models
    (Decision Tree, Random Forest, Adaboost, XGBoost, and a Neural Network) at each taxonomic parent node of the dataset.
    This forms five cascading taxonomic classifiers, with a model at each taxonomic parent node.
    This enables comparison of the models at each taxonomic level, to determine the most robust and optimal model to use
    as a metadata classifier.

    Attributes:
        model_abbreviations (dict): A dictionary containing the names of the classification models as keys, and their abbreviations as values.
        model_save_types (dict): A dictionary containing the names of the classification models as keys, and their relevant file types when saved.
        file_name_taxon (dict): A dictionary containing the taxonomic level indicators in the dataset, and their relevant abbreviations to be used in file naming.
"""

import pandas as pd

import pipelines
import decision_tree
import random_forest
import xgboost_model
import neural_network_model
import adaboost_model
from pipelines import sub_species_detection
import silhouette_k_means

model_abbreviations = {'Neural network': 'nn',
                       'Decision tree': 'dt',
                       'Random forest': 'rf',
                       'Xgboost': 'xgb',
                       'AdaBoost': 'ada'}

model_save_types = {'Neural network': '',
                    'Decision tree': '.sav',
                    'Random forest': '.sav',
                    'Xgboost': '.json',
                    'AdaBoost': '.sav'}

file_name_taxon = {'taxon_family_name': '_family',
                   'taxon_genus_name': '_genus',
                   'taxon_species_name': '_species',
                   'sub_species': '_subspecies'}


def dataset_iteration(observation_file: str, metadata_file: str):
    """This method is performs the full metadata training for all models at all available taxonomic levels for the
    provided dataset. Only a single dataset is trained at a time


    The information printed out, is to be used within the `model_comparison.ipynb` to direct the model validation and figure construction.
    For more information, please review the `model_comparison` notebook.

    Args:
        observation_file (str): The name of the observation files within the `data/processed/` directory.
        metadata_file (str): The name of the metadata files within the `data/processed/` directory. Note they must coincide with the order of the observation files.
    """
    models, model_name_collection, taxon_target_collection, abbreviations_collection = model_iteration(observation_file, metadata_file)

    # Information to be used in the model_comparison.ipynb notebook
    print('Models: ', models)
    print('File names: ', model_name_collection)
    print('Taxon targets: ', taxon_target_collection)
    print('Model abbreviations: ', abbreviations_collection)


# Train all models on the provided dataset.
def model_iteration(observation_file: str, metadata_file: str):
    """Method performs the model iteration per dataset.

    Args:
        observation_file (str): The processed iNaturalist observations dataset.
        metadata_file (str): The corresponding metadata for the observation file.

    Returns:
        models (list): The list of all models iterated over.
        model_name_collection (list): The list of all model names produced during the iteration of the dataset.
        taxon_target_collection (list): The list of all taxonomic targets iterated through within this dataset.
        abbreviations_collection (list): A list of the corresponding model abbreviations
    """
    # Multi-model collections
    models = list(model_abbreviations.keys())

    model_name_collection = []  # Collections across multiple model collections (this is the same for all models)
    taxon_target_collection = []
    abbreviations_collection = []

    taxon_models = []  # Collections across single models
    taxon_targets = []

    for model in models:  # Iterate across all models
        taxon_models, taxon_targets = taxonomic_level_modelling(observation_file, metadata_file, model)  # Iterate across all taxonomic levels per model

        abbreviations_collection.append(model_abbreviations[model])  # Collect essential information regarding file names for future use

    model_name_collection.append(taxon_models)  # Collect model file names which indicate taxonomic level (supplemented by model abbreviations)
    taxon_target_collection.append(taxon_targets)  # Collect taxonomic target levels

    return models, model_name_collection, taxon_target_collection, abbreviations_collection  # Return gathered file information


def taxonomic_level_modelling(observation_file: str, metadata_file: str, model: str):
    """This method performs a taxonomic level breakdown and training at all taxonomic levels for the specified model.

    This method performs the dataset taxonomic restriction at the parent node, modifying the dataset to fit each taxonomic parent node,
    such that only the taxonomic children of the parent node are within the dataset. This is done for the entire taxonomic structure within the dataset.

    Args:
        observation_file (str): The processed iNaturalist observations dataset.
        metadata_file (str): The corresponding metadata for the observation file.
        model (str): String specification of the model to be trained ane evaluated.

    Returns:
        models (list): A list of file names, where the file name specified the taxonomic parent node (model classifies the taxonomic children)
        taxon_targets (list): Species the list of the taxonomic target levels in the same order as the models list.
    """
    # Collection of taxon level information for notebook use
    models = []  # Collection of model file names
    taxon_targets = []  # Collection of taxonomic target levels.

    df_prime = pipelines.aggregate_data(observation_file, metadata_file)  # Aggregate observations with metadata

    df_prime = df_prime[df_prime['taxon_species_name'] != 'Felis catus']  # Remove common household cat from dataset

    df_prime = df_prime.apply(lambda x: sub_species_detection(x), axis=1) # Extract subspecies labels

    taxon_breakdown = taxonomic_analysis(df_prime.copy())  # Generate a taxonomic breakdown (dictionary with taxon level as the keys, and a list of taxonomic labels as the values)
    taxonomic_keys = list(taxon_breakdown.keys())  # Extract taxonomic levels (keys)

    print('<----------->')
    print('Taxon breakdown: ', taxon_breakdown)

    for i in range(len(taxonomic_keys) - 1):  # Iterate through taxonomic keys until species (taxonomic parent node level)
        if len(taxon_breakdown[taxonomic_keys[i + 1]]) > 1:  # Ensure there are more than a single child in the taxonomic level below
            taxon_parent_level = taxonomic_keys[i]

            # Restriction
            for restriction in taxon_breakdown[taxonomic_keys[i]]:  # Restrict the dataset to one of the labels at the parent note
                df = df_prime.copy()

                df = df[df[taxonomic_keys[i]] == restriction]  # Enforce taxon restriction

                target_taxon = taxonomic_keys[i + 1]  # Extract the target taxon (taxonomic child level)
                df = df.dropna(subset=[target_taxon])  # Remove any NaN labels at this taxonomic level

                # Taxonomic level clean-up to determine number of classes (with restriction)
                df = df.dropna(subset=['public_positional_accuracy'])  # Remove n/a entries
                df = df[df['public_positional_accuracy'] <= 40000]  # Remove entries with inadequate accuracy
                df = df[df.groupby(target_taxon).common_name.transform('count') >= 5].copy()  # Enforce at least 10 observations

                if df[target_taxon].nunique() <= 1:  # Check at least two classes present with restriction
                    continue

                file_start = generate_file_name_start(restriction)  # Generate file_start_name based on the parent taxon level and the restriction (parent node)

                # Print Information
                print('------------------------------')
                print("Taxonomic Parent level: ", taxon_parent_level)
                print("Restriction: ", restriction)
                print('Target taxon: ', target_taxon)
                print('Model: ', model)

                # Execute the model training with the restricted and processed data
                model_simplification(df=df,
                                     model=model,
                                     target_taxon=target_taxon,
                                     model_save_type=model_save_types[model],
                                     file_name_start=file_start)
                models.append(file_start)  # Save the model file name for notebook input
                taxon_targets.append(target_taxon)
    return models, taxon_targets


def taxonomic_analysis(df: pd.DataFrame):
    """This method performs the taxonomic breakdown of the dataset at the following taxonomic levels: taxon_family_name, taxon_genus_name, taxon_species_name, subspecies

    Args:
        df (DataFrame): The dataframe containing the unrestricted observations and metadata to perform a taxonomic breakdown of the entire dataset

    Returns:
        (dict): Keys specify the taxonomic level and the values are a list containing all unique labels in the dataset, forming a taxonomic breakdown
    """
    taxonomy_list = ['taxon_family_name', 'taxon_genus_name', 'taxon_species_name', 'sub_species']   # Taxonomic levels to target in breakdown
    taxon_breakdown = dict()  # Create an empty dictionary

    for taxon in taxonomy_list:  # Iterate through the taxonomic levels
        df = df.dropna(subset=[taxon])  # Remove all n/a labels
        taxon_breakdown[taxon] = df[taxon].unique().tolist()  # Find unique taxon level labels. These form the values to the taxonomic key in the dictionary
    return taxon_breakdown  # Return the taxonomic breakdown dictionary


# Method generates unique file name that can be referenced for evaluation in the notebook
def generate_file_name_start(restriction: str):
    """Method standardizes the parent taxonomic restriction to create a suitable filename for each model

        This method removes white space, replacing it with an underscore, and ensures the name is all lower case.
        Args:
            restriction (str): The label of the taxonomic parent node (restriction)

        Returns:
            (str): A standardized form of the restriction.
    """
    restriction = restriction.replace(" ", "_")
    restriction = restriction.lower()
    return restriction


def model_simplification(df: pd.DataFrame,
                         model: str,
                         target_taxon,
                         model_save_type: str,
                         file_name_start: str):
    """This method simplifies the model training, testing, and saving process. It completes a full model training,
    testing, and saving for the specified model and dataset

    Args:
        df (DataFrame): The combined observation and metadata dataframe with the taxonomic parent node restriction applied. Only taxonomic child labels are present in df.
        model (str): Specification of the model to be used to classify the taxonomic child nodes.
        target_taxon (str): Specification of the taxonomic level of the taxon child nodes (not the taxonomic level of the parent node.
        model_save_type (str): Specification of the model file type/ model suffix.
        file_name_start (str): The standardized taxon parent node label which will be used to construct a unique file name for the trained model.
    """

    model_abbr = model_abbreviations[model]  # Get model abbreviation

    model_name = file_name_start + "_" + model_abbr + "_model" + model_save_type  # Put together essential file names
    training_history = file_name_start + "_" + model_abbr + '_training_accuracy.csv'
    validation_file = file_name_start + "_" + model_abbr + '_validation.csv'

    model_selection_execution(model, df, target_taxon, model_name, training_history, validation_file)  # Select model, and execute the required process (data pipeline, model training, and evaluation)


# Method allows for multiple model selection, training, evaluation, and saving
def model_selection_execution(model: str,
                              df: pd.DataFrame,
                              target_taxon: str,
                              model_name: str,
                              training_history: str,
                              validation_file: str):
    """This method allows multiple models to be trained through the specification of the model type, and the subsequent execution of the required data pipeline.

    Args:
        model (str): Specification of the model to be used to classify the taxonomic child nodes.
        df (DataFrame): The combined observation and metadata dataframe with the taxonomic parent node restriction applied. Only taxonomic child labels are present in df.
        target_taxon (str): Specification of the taxonomic level of the taxon child nodes (not the taxonomic level of the parent node)
        model_name (str): The complete model name (parent taxon label and model abbreviation make the combined name unique)
        training_history (str): File name at which to save the model training history.
        validation_file (str): File name at which to save the validation dataset.
    Returns:
        (None)
    """
    match model:
        case 'Neural network':
            return neural_network_model.neural_network_process(df, target_taxon, model_name,
                                                               training_history, validation_file)
        case 'Decision tree':
            return decision_tree.decision_tree_process(df, target_taxon, model_name, training_history,
                                                       validation_file)
        case 'Random forest':
            return random_forest.random_forest_process(df, target_taxon, model_name, training_history,
                                                       validation_file)
        case 'Xgboost':
            return xgboost_model.xgboost_process(df, target_taxon, model_name, training_history,
                                                 validation_file)
        case 'AdaBoost':
            return adaboost_model.adaboost_process(df, target_taxon, model_name, training_history,
                                                   validation_file)


def train_base_model(model: str, target_taxon, file_name='base_meta'):
    """This method trains the root node of the taxonomic tree.

    The current model training requires the Felid and Elephant datasets to be kept separate to train all of their relevant taxonomic models.
    This however excludes the root classifier to determine between the two taxon families.
    This method ensures the taxonomic root is trained.
    Note, this method can be used to train a metadata global classifier by specifying the target taxom to the species level.

    Args:
        model (str): Specification of the model to be used to classify the taxonomic child nodes.
        target_taxon (str): Specification of the taxonomic level of the taxon child nodes (not the taxonomic level of the parent node)
        file_name (str): The file name of the root classification model.
    """
    df_felids = pipelines.aggregate_data('felids_train.csv', 'felids_meta.csv')  # Aggregate felid observations and metadata
    df_proboscidia = pipelines.aggregate_data('proboscidia_train.csv', 'proboscidia_meta.csv')  # Aggregate elephant observations and metadata
    df = pd.concat([df_felids, df_proboscidia])  # Joint both felid and elephant datasets into one

    silhouette_k_means.k_max = 84  # Modify silhouette score process to increase the range and interval due to the large amount of datapoints.
    silhouette_k_means.k_interval = 20

    model_suffix = model_abbreviations[model]  # Get model suffix
    model_selection_execution(model,
                              df,
                              target_taxon,
                              file_name + '_model' + model_suffix,
                              file_name + '_training_accuracy',
                              file_name + '_validation.csv')  # Train model on one model type at a time


if __name__ == '__main__':
    """This method executes the training of the root classifier or the full taxonomic tree of the specified dataset. 
    
    The root flag when set to True will perform the root classifier training. Changing this to false, allows for the full taxonomic tree classification 
    training at each parent node (of the provided dataset).
    """
    root = True

    if root:
        train_base_model('Xgboost', 'taxon_family_name')
    else:
        dataset_iteration(observation_file='proboscidia_train.csv', metadata_file='proboscidia_meta.csv')
