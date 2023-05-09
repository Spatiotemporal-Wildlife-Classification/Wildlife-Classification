import sys
import random
import numpy as np
from src.structure import Config
from src.models.meta.pipelines import sub_species_detection
import pandas as pd
import os
import shutil

raw_img_path = Config.root_dir() + '/data/taxon_raw/'
img_path = Config.root_dir() + '/data/taxon/'
test_path = Config.root_dir() + '/data/taxon_test/'
data_path = Config.root_dir() + '/data/processed/'
multiple_detections_id = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r']

img_size = 528
batch_size = 32
test_split = 0.15

taxonomy_list = ['taxon_family_name', 'taxon_genus_name', 'taxon_species_name', 'sub_species']
count = 0
length = 0


def create_dataset(observations: list):
    global length
    df_obs = pd.DataFrame()

    for observation in observations:
        df_obs = pd.concat([df_obs, pd.read_csv(data_path + observation)])

    df_obs = df_obs.dropna(subset=['image_url'])

    df_obs = df_obs[df_obs['taxon_species_name'] != 'Felis catus']

    # Remove unnecessary columns
    df_obs = df_obs.drop(columns=['observed_on', 'local_time_observed_at', 'positional_accuracy'])
    length = len(df_obs)
    return df_obs


def sub_species_detection(x):
    name_count = len(x['scientific_name'].split())
    x['sub_species'] = np.nan
    if name_count >= 3:
        x['sub_species'] = x['scientific_name']
    return x


def taxonomic_analysis(df: pd.DataFrame):
    # Taxonomy breakdown
    taxonomy_list = ['taxon_family_name', 'taxon_genus_name', 'taxon_species_name', 'sub_species']
    taxon_breakdown = dict()

    # Identify each unique taxon level mammal
    for taxon in taxonomy_list:
        df = df.dropna(subset=[taxon])  # Remove all n/a labels
        taxon_breakdown[taxon] = df[taxon].unique().tolist()  # Find unique taxon level genus names
    return taxon_breakdown


def image_download(x):
    global count
    path = img_path
    set_decider = random.uniform(0, 1)
    if set_decider < test_split:
        path = test_path

    for level in taxonomy_list:
        taxon_level = x[level]

        if taxon_level is np.nan:
            break

        # Clean file path
        taxon_level = taxon_level.replace(" ", "_")
        taxon_level = taxon_level.lower()
        path = path + taxon_level + "/"

    multiple_obs(x['id'], path)

    count = count + 1
    status_bar_update()


def multiple_obs(id, path):
    for suffix in multiple_detections_id:
        raw_path = raw_img_path + str(id) + '_' + suffix + '.jpg'
        if os.path.exists(raw_path):
            file_name = path + str(id) + '_' + suffix + '.jpg'
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            shutil.move(raw_path, file_name)
        else:
            break


def status_bar_update():
    progress_bar_length = 50
    percentage_complete = count / length
    filled = int(progress_bar_length * percentage_complete)

    bar = '=' * filled + '-' * (progress_bar_length - filled)
    percentage_display = round(100 * percentage_complete, 5)
    sys.stdout.write('\r[%s] %s%s ... count: %s' % (bar, percentage_display, '%', count))
    sys.stdout.flush()


if __name__ == "__main__":
    observations = ['proboscidia_final.csv', 'felids_final.csv']
    df = create_dataset(observations)

    # Generate sub_species
    df = df.apply(lambda x: sub_species_detection(x), axis=1)

    taxon_breakdown = taxonomic_analysis(df.copy())

    df.head(24000).apply(lambda x: image_download(x), axis=1)
