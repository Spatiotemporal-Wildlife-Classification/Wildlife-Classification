import pandas as pd

import pipelines
import decision_tree
import neural_network
import random_forest
import xgboost_model
import adaboost_model
from pipelines import sub_species_detection

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


def dataset_iterations(observation_files: list, metadata_files: list, k_centroids: list):
    for (observation_file, metadata_file, k_centroid) in zip(observation_files, metadata_files, k_centroids):
        models, model_name_collection, taxon_target_collection, abbreviations_collection = model_iteration(
            observation_file, metadata_file, k_centroid)

    print('Models: ', models)
    print('File names: ', model_name_collection)
    print('Taxon targets: ', taxon_target_collection)
    print('Model abbreviations: ', abbreviations_collection)


def model_iteration(observation_file: str,
                    metadata_file: str,
                    k_centroids: int):
    # Multi-model collections
    models = list(model_abbreviations.keys())
    model_name_collection = []
    taxon_target_collection = []
    abbreviations_collection = []

    # Single model collections
    taxon_models = []
    taxon_targets = []

    for model in models:
        taxon_models, taxon_targets = taxonomic_level_modelling(observation_file,
                                                                metadata_file,
                                                                model,
                                                                k_centroids)

        abbreviations_collection.append(model_abbreviations[model])

    model_name_collection.append(taxon_models)
    taxon_target_collection.append(taxon_targets)

    # Essential information
    return models, model_name_collection, taxon_target_collection, abbreviations_collection


def taxonomic_level_modelling(observation_file: str,
                              metadata_file: str,
                              model: str,
                              k_centroids: int):
    # Collection of info for Notebook
    models = []
    taxon_targets = []

    # Data aggregation
    df_prime = pipelines.aggregate_data(observation_file, metadata_file)

    # Remove dominant species
    df_prime = df_prime[df_prime['taxon_species_name'] != 'Felis catus']

    # Generate sub_species
    df_prime = df_prime.apply(lambda x: sub_species_detection(x), axis=1)

    # Taxon breakdown
    taxon_breakdown = taxonomic_analysis(df_prime.copy())
    taxonomic_keys = list(taxon_breakdown.keys())

    print('<----------->')
    print('Taxon breakdown: ', taxon_breakdown)

    for i in range(len(taxonomic_keys) - 1):
        if len(taxon_breakdown[taxonomic_keys[i + 1]]) > 1:
            taxon_parent_level = taxonomic_keys[i]

            # Restriction
            for restriction in taxon_breakdown[taxonomic_keys[i]]:
                df = df_prime.copy()

                # Enforce taxon restriction
                df = df[df[taxonomic_keys[i]] == restriction]

                # Target taxon
                target_taxon = taxonomic_keys[i + 1]
                df = df.dropna(subset=[target_taxon])

                # Check at least two classes present with restriction
                df = df.dropna(subset=['public_positional_accuracy'])
                df = df[df['public_positional_accuracy'] <= 40000]
                df = df[df.groupby(target_taxon).common_name.transform('count') >= 10].copy()
                if df[target_taxon].nunique() <= 1:
                    continue

                # Generate file_start_name
                file_start = generate_file_name_start(taxon_parent_level, restriction)

                # Print Information
                print('------------------------------')
                print("Taxonomic Parent level: ", taxon_parent_level)
                print("Restriction: ", restriction)
                print('Target taxon: ', target_taxon)
                print('Model: ', model)

                # Data pipeline process
                model_simplification(df=df,
                                     model=model,
                                     target_taxon=target_taxon,
                                     k_centroids=k_centroids,
                                     model_save_type=model_save_types[model],
                                     file_name_start=file_start)
                models.append(file_start)
                taxon_targets.append(target_taxon)
    return models, taxon_targets


def taxonomic_analysis(df: pd.DataFrame):
    # Taxonomy breakdown
    taxonomy_list = ['taxon_family_name', 'taxon_genus_name', 'taxon_species_name', 'sub_species']
    taxon_breakdown = dict()

    for taxon in taxonomy_list:
        df = df.dropna(subset=[taxon])
        taxon_breakdown[taxon] = df[taxon].unique().tolist()
    return taxon_breakdown


def generate_file_name_start(parent_taxon: str, restriction: str):
    taxon = file_name_taxon[parent_taxon]
    restriction = restriction.replace(" ", "_")
    return restriction + taxon


def model_simplification(df: pd.DataFrame,
                         model: str,
                         target_taxon,
                         k_centroids: int,
                         model_save_type: str,
                         file_name_start: str):
    # Get model abbreviation
    model_abbr = model_abbreviations[model]

    # Put together essential file names
    model_name = file_name_start + "_" + model_abbr + "_model" + model_save_type
    training_history = file_name_start + "_" + model_abbr + '_training_accuracy.csv'
    validation_file = file_name_start + "_" + model_abbr + '_validation.csv'

    model_selection_execution(model, df, target_taxon, k_centroids, model_name, training_history, validation_file)


def model_selection_execution(model: str,
                              df: pd.DataFrame,
                              target_taxon: str,
                              k_centroids: int,
                              model_name: str,
                              training_history: str,
                              validation_file: str):
    match model:
        case 'Neural network':
            return neural_network.neural_network_process(df, target_taxon, k_centroids, model_name, training_history,
                                                         validation_file)
        case 'Decision tree':
            return decision_tree.decision_tree_process(df, target_taxon, k_centroids, model_name, training_history,
                                                       validation_file)
        case 'Random forest':
            return random_forest.random_forest_process(df, target_taxon, k_centroids, model_name, training_history,
                                                       validation_file)
        case 'Xgboost':
            return xgboost_model.xgboost_process(df, target_taxon, k_centroids, model_name, training_history,
                                                 validation_file)
        case 'AdaBoost':
            return adaboost_model.adaboost_process(df, target_taxon, k_centroids, model_name, training_history,
                                                   validation_file)


if __name__ == '__main__':
    dataset_iterations(observation_files=['proboscidia_final.csv', 'felids_final.csv'],
                       metadata_files=['proboscidia_meta.csv', 'felids_meta.csv'],
                       k_centroids=[20, 40])
