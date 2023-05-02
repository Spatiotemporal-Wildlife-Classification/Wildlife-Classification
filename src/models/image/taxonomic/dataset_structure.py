import sys
from datetime import datetime

import numpy as np
import requests
from keras.utils import image_dataset_from_directory
from src.structure import Config
from src.models.meta.pipelines import sub_species_detection
import pandas as pd
import os

# img_path = os.path.join(os.getcwd(), 'data', 'taxon')
img_path = Config.root_dir() + '/data/taxon/'
data_path = Config.root_dir() + '/data/processed/'
img_size = 528
batch_size = 32


taxonomy_list = ['taxon_family_name', 'taxon_genus_name', 'taxon_species_name', 'sub_species']
count = 0
length = 0


def create_datasets(observations: str):
    global length
    df_obs = pd.read_csv(data_path + observations)

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
    for level in taxonomy_list:
        taxon_level = x[level]

        if taxon_level is np.nan:
            break

        # Clean file path
        taxon_level = taxon_level.replace(" ", "_")
        taxon_level = taxon_level.lower()
        path = path + taxon_level + "/"

    # Download image
    file_name = path + str(x['id']) + '.jpg'
    if not os.path.exists(file_name):
        try:
            img_data = requests.get(x['image_url'], timeout=3).content

            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            with open(file_name, "wb") as f:
                f.write(img_data)
        except:
            print('Error in retrieving image')

    count = count + 1
    status_bar_update()


def status_bar_update():
    progress_bar_length = 50
    percentage_complete = count / length
    filled = int(progress_bar_length * percentage_complete)

    bar = '=' * filled + '-' * (progress_bar_length - filled)
    percentage_display = round(100 * percentage_complete, 5)
    sys.stdout.write('\r[%s] %s%s ... count: %s' % (bar, percentage_display, '%', count))
    sys.stdout.flush()


if __name__ == "__main__":
    observations = 'proboscidia_final.csv'
    df = create_datasets(observations)

    # Generate sub_species
    df = df.apply(lambda x: sub_species_detection(x), axis=1)

    taxon_breakdown = taxonomic_analysis(df.copy())

    df.head(15000).apply(lambda x: image_download(x), axis=1)
