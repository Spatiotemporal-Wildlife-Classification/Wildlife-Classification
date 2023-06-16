"""This file establishes the dynamic pipelines used to produce training, test, and validation sets from the dataset.

   A dynamic pipeline accepts processed data, and transforms it into: training data, test data, validation data with
   the specified labels. The pipelines account for the variable taxonomic levels and the encoding of the location
   feature, to produce the above transformations.

   Note, the encoding of the location feature occurs within the pipeline processes. Please review the Silhouette
   score documentation for further information on the process.

   Attributes:
       root_path (str): The path to the project root.
       data_path (str): The path to where the data is stored within the project
       save_path (str): The path to where models and validation data (if created) is saved. To train the models used in ensemble use `/models/meta/`. To metamodel notebook comparison use `/notebooks/meta_modelling/model_comparison_cache/`
       validation_set_flag (bool): A boolean flag indicating whether a validation set should be created and saved. The validation set is saved to save_path. Each file will have suffixx `_validation.csv`
"""
from sklearn.cluster import KMeans
# Modelling
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from imblearn.over_sampling import RandomOverSampler

# General
import numpy as np
import pytz
from pandas.api.types import CategoricalDtype
import pandas as pd

# Geolocation libraries
from global_land_mask import globe

# Config and local
from src.structure.Config import root_dir
import src.models.meta.silhouette_k_means as silhouette_k_means

# Paths
root_path = root_dir()  # Root path of the project
data_path = '/data/obs_and_meta/processed/'  # Path to where data is stored
save_path = '/models/meta/'  # '/notebooks/meta_modelling/model_comparison_cache/' to produce models for evaluation within the meta_data_model_comparison notebook

# Boolean Flags
validation_set_flag = False


def decision_tree_data(df: pd.DataFrame, taxon_target: str, validation_file: str):
    """Method to create the train/set/validation data to be used by the decision tree/ random forest/ Adaboost models

    Args:
        df (DataFrame): The dataframe containing all data for each observation.
        taxon_target (str): The taxonomic target level, to extract the correct labels (taxon_family_name, taxon_genus_name, taxon_species_name, subspecies)
        validation_file (str): The name of the file where the validation data will be stored. Also informs the name of the saved models.

    Returns:
        X (DataFrame): The features in a form suitable for direct use within the models.
        y (Series): The labels for the corresponding observations as the correct taxonomic level.
    """
    k_means = silhouette_k_means.silhouette_process(df, validation_file)
    X, y = tree_pipeline(df, k_means, taxon_target, validation_file)
    return X, y


def xgb_data(df: pd.DataFrame, taxon_target: str, validation_file: str):
    """Method to create the train/set/validation data to be used by the XGBoost model.

    Args:
        df (DataFrame): The dataframe containing all data for each observation.
        taxon_target (str): The taxonomic target level, to extract the correct labels (taxon_family_name, taxon_genus_name, taxon_species_name, subspecies)
        validation_file (str): The name of the file where the validation data will be stored. Also informs the name of the saved models.

    Returns:
        X (DataFrame): The features in a form suitable for direct use within the models.
        y (Series): The labels for the corresponding observations as the correct taxonomic level.
    """
    k_means = silhouette_k_means.silhouette_process(df, validation_file)
    X, y = xgb_pipeline(df, k_means, taxon_target, validation_file)
    return X, y


def neural_network_data(df: pd.DataFrame, taxon_target: str, validation_file: str):
    """Method to create the train/set/validation data to be used by the neural network model.

    Args:
        df (DataFrame): The dataframe containing all data for each observation.
        taxon_target (str): The taxonomic target level, to extract the correct labels (taxon_family_name, taxon_genus_name, taxon_species_name, subspecies)
        validation_file (str): The name of the file where the validation data will be stored. Also informs the name of the saved models.

    Returns:
        X (DataFrame): The features in a form suitable for direct use within the models.
        y (Series): The labels for the corresponding observations as the correct taxonomic level.
        classes (int): The number of classes data labels
    """
    k_means = silhouette_k_means.silhouette_process(df, validation_file)
    X, y, lb, classes = nn_pipeline(df, k_means, taxon_target, validation_file)
    return X, y, classes


def general_pipeline(df: pd.DataFrame, k_means: KMeans, taxon_target: str):
    """Method performs general pipeline functions for all model types (Neural network, XGBoost, AdaBoost, Decision tree, Random Forest)

    Args:
        df (DataFrame): The dataframe containing all observation data from the processed data directory.
        k_means (KMeans): The trained K-means model that performs the location encoding
        taxon_target (str): The taxonomic level at which to extract the taxon labels (taxon_family_name, taxon_genus_name, taxon_species_name, sub_species)

    Returns:
        (DataFrame): A dataframe containing cleaned, transformed, and new data features for further specified processing depending on the model.
    """
    # Data Cleaning
    df = df.drop(columns=['geoprivacy', 'taxon_geoprivacy', 'taxon_id', 'license', 'image_url'])  # Remove non-essential columns
    df = df.dropna(subset=['taxon_species_name', 'public_positional_accuracy'])  # Remove null species names and positional accuracies
    df = df[df['public_positional_accuracy'] <= 40000]  # Positional Accuracy Restriction
    df = df.drop(columns=['public_positional_accuracy'])  # Drop the public positional accuracy column

    # Transformations
    df = df.apply(lambda x: sub_species_detection(x), axis=1)  # Generate subspecies labels
    df = df.drop(columns=['scientific_name'])  # Drop the scientific name column

    df = df[df.groupby(taxon_target).common_name.transform('count') >= 5].copy()  # Remove species with less than 5 observations

    df['location_cluster'] = k_means.predict(df[['latitude', 'longitude']])  # Location encoding using K-means

    df['land'] = 1  # All observations from dataset are terrestrial. For unknown datasets use the `land_mask()` method to automate the feature value

    df = df.apply(lambda x: elevation_clean(x), axis=1)  # Clean elevation values. In aquatic observations, the max elevation is sea level 0m
    df['elevation'] = df['elevation'].fillna(df.groupby('taxon_species_name')['elevation'].transform('mean'))  # If elevation is missing, interpolate with mean species elevation

    df['hemisphere'] = (df['latitude'] >= 0).astype(int)  # Northern/ Southern hemisphere feature
    df = df.drop(columns=['latitude', 'longitude']) # Remove longitude and latitude columns

    df['observed_on'] = pd.to_datetime(df['observed_on'], format="%Y-%m-%d %H:%M:%S%z", utc=True)  # Datetime transform into datetime object
    df['month'] = df['observed_on'].dt.month  # Month feature
    df['hour'] = df.apply(lambda x: x['observed_on'].astimezone(pytz.timezone(x['time_zone'])).hour, axis=1)  # Local time zone hour feature
    df = day_night_calculation(df)  # Day/ night feature
    df = df.apply(lambda x: season_calc(x), axis=1)  # Season feature into categorical values
    df = ohe_season(df)  # One-hot-encode the categorical season values

    return df


def tree_pipeline(df, k_means, taxon_target, validation_file: str):
    """This method performs further data processing to structure and format it for use in a decision tree, random forest and adaboost models.

    Args:
        df (DataFrame): The dataframe containing all observation data from the processed data directory.
        k_means (KMeans): The trained K-means model that performs the location encoding
        taxon_target (str): The taxonomic level at which to extract the taxon labels (taxon_family_name, taxon_genus_name, taxon_species_name, sub_species)
        validation_file (str): The name of file to store validation data. Informs model naming as well.

    Returns:
        X (DataFrame): A dataframe containing features in rows and observations in column ready for use as input features to the models for training and evaluation.
        y (Series): The categorical labels of the associated observations at the correct taxonomic level specified.
    """
    df = general_pipeline(df, k_means, taxon_target)  # Perform general pipeline

    df = df.drop(columns=['observed_on', 'time_zone'])  # Drop observed on column as date & time transformations are complete

    df = validation_set(df, taxon_target, validation_file)  # Create validation set for further testing

    # Data formatting
    taxon_y = df[taxon_target]  # Retrieve labels at taxonomic target level

    if taxon_y.isnull().any():  # If no taxonomic label is present, remove the observation
        df = df.dropna(subset=[taxon_target])

    y = df[taxon_target]  # Extract labels
    X = df.drop(columns=['taxon_kingdom_name', 'taxon_phylum_name',
                         'taxon_class_name', 'taxon_order_name', 'taxon_family_name',
                         'taxon_genus_name', 'taxon_species_name', 'sub_species', 'common_name'])  # Extract features only

    X, y = over_sample(X, y)  # Resample dataset to reduce imbalance
    return X, y


def xgb_pipeline(df, k_means, taxon_target, validation_file: str):
    """This method performs further data processing to structure and format it for use in the XGBoost model

    This method makes use of the decison_tree_pipeline, simply encoding the labels in a One-Hot-Encoded (OHE) format

    Args:
        df (DataFrame): The dataframe containing all observation data from the processed data directory.
        k_means (KMeans): The trained K-means model that performs the location encoding
        taxon_target (str): The taxonomic level at which to extract the taxon labels (taxon_family_name, taxon_genus_name, taxon_species_name, sub_species)
        validation_file (str): The name of file to store validation data. Informs model naming as well.

    Returns:
        X (DataFrame): A dataframe containing features in rows and observations in column ready for use as input features to the models for training and evaluation.
        y (Series): The OHE encoding of the observation labels at the correct taxonomic level specified.
    """
    X, y = tree_pipeline(df, k_means, taxon_target, validation_file)

    y, classes = ohe_labels(y)  # OHE labels

    return X, y


def nn_pipeline(df, k_means, taxon_target, validation_file: str):
    """This method performs further data processing to structure and format it for use in the Neural Network model

    This method performs similar processing steps to both the decision tree and XGBoost pipelines.
    However, categorical variables are required to be OHE and the resulting features are normalized for use in the model.

    Args:
        df (DataFrame): The dataframe containing all observation data from the processed data directory.
        k_means (KMeans): The trained K-means model that performs the location encoding
        taxon_target (str): The taxonomic level at which to extract the taxon labels (taxon_family_name, taxon_genus_name, taxon_species_name, sub_species)
        validation_file (str): The name of file to store validation data. Informs model naming as well.

    Returns:
        X (DataFrame): A dataframe containing features in rows and observations in column ready for use as input features to the models for training and evaluation. These features are normalized.
        y (Series): The OHE encoding of the observation labels at the correct taxonomic level specified.
    """
    df = general_pipeline(df, k_means, taxon_target)

    # Generate dummy variables for categorical features
    df = pd.get_dummies(df, prefix='loc', columns=['location_cluster'], drop_first=True)  # OHE location cluster feature
    df = pd.get_dummies(df, prefix='hr', columns=['hour'], drop_first=True)  # OHE hour feature
    df = pd.get_dummies(df, prefix='mnth', columns=['month'], drop_first=True)  # OHE month feature

    df = df.drop(columns=['observed_on', 'time_zone'])  # Drop observed on column as date & time transformations are complete
    df = validation_set(df, taxon_target, validation_file)  # Create validation set for further testing

    # Data formatting
    taxon_y = df[taxon_target]  # Retrieve labels at taxonomic target level

    if taxon_y.isnull().any():  # If no taxonomic label is present, remove the observation
        df = df.dropna(subset=[taxon_target])

    y = df[taxon_target]  # Extract labels
    X = df.drop(columns=['taxon_kingdom_name', 'taxon_phylum_name',
                         'taxon_class_name', 'taxon_order_name', 'taxon_family_name',
                         'taxon_genus_name', 'taxon_species_name', 'sub_species', 'common_name'])  # Extract features only

    X, y = over_sample(X, y)  # Resample dataset to reduce imbalance

    y, classes = ohe_labels(y)  # OHE labels

    # Normalize data using min-max approach
    norm_columns = ['apparent_temperature', 'apparent_temperature_max', 'apparent_temperature_min',
                    'cloudcover', 'cloudcover_high', 'cloudcover_low', 'cloudcover_mid', 'dewpoint_2m',
                    'diffuse_radiation', 'direct_radiation', 'elevation', 'et0_fao_evapotranspiration_daily',
                    'et0_fao_evapotranspiration_hourly', 'precipitation', 'precipitation_hours',
                    'precipitation_sum', 'rain', 'rain_sum', 'relativehumidity_2m', 'shortwave_radiation',
                    'shortwave_radiation_sum', 'snowfall', 'snowfall_sum', 'soil_moisture_0_to_7cm',
                    'soil_moisture_28_to_100cm', 'soil_moisture_7_to_28cm', 'soil_temperature_0_to_7cm',
                    'soil_temperature_28_to_100cm', 'soil_temperature_7_to_28cm', 'surface_pressure',
                    'temperature_2m', 'temperature_2m_max', 'temperature_2m_min', 'vapor_pressure_deficit',
                    'winddirection_100m', 'winddirection_10m', 'winddirection_10m_dominant',
                    'windgusts_10m', 'windgusts_10m_max', 'windspeed_100m', 'windspeed_10m',
                    'windspeed_10m_max']
    X[norm_columns] = StandardScaler().fit_transform(X[norm_columns])
    return X, y, classes


def ohe_labels(y):
    """This method encodes the taxonomic labels in a One-hot-encoded format.

    Special consideration is enforced for binary labels such that the resulting ohe labels are of the form [0, 1] or [1, 0]

    Args:
        y (Series): The categorical taxonomic labels

    Returns:
        (Series): OHE taxonomic labels
    """
    classes = y.nunique()  # OHE encode the labels
    lb = LabelBinarizer()
    lb.fit(y)
    y = lb.transform(y)

    if classes == 2:  # Modification required if only two classes are present
        y = nn_binary_label_handling(y)
    return y, classes


def validation_set(df: pd.DataFrame, taxon_target: str, file_name: str):
    """This method creates a validation set from the provided dataframe for further model evaluation

    The validation set comprises 20% of each class's composition from the dataframe.
    The observations included in the validation set are removed from the dataframe.

    Args:
        df (DataFrame): The dataframe containing all observation data from the processed data directory.
        taxon_target (str): The taxonomic level at which to extract the taxon labels (taxon_family_name, taxon_genus_name, taxon_species_name, sub_species)
        file_name (str): The name of the file in which the validation data will be stored.

    Returns:
        (DataFrame): The dataframe with the validation observations removed
    """
    if validation_set_flag:
        grouped = df.groupby([taxon_target]).sample(frac=0.2, random_state=2)  # 20% of each class goes to the validation set
        grouped.to_csv(root_path + save_path + file_name)  # Save evaluation dataset

        df = df.drop(grouped.index)  # Remove validation observations from the current df through their index
    return df


def over_sample(X, y):
    """This method performs oversampling on the dataset in order to provide a more balanced data distribution, to combat the tail-end distribution (characteristic of wildlife data).

    Note, the oversampling aimed to increase the quantity of observations in minority classes to achieve a more even distribution.

    Args:
        X (DataFrame): The dataset's observation features to be used in model training and evaluation.
        y (Series): The label for each observation (still categorical)

    Returns:
         X_res (DataFrame): The features dataset with additional observations due to the oversampling
         y_res (Series): An associated dataframe containing the observation labels, including for the additional observations created.
    """
    ros = RandomOverSampler(sampling_strategy='minority', random_state=2)
    X_res, y_res = ros.fit_resample(X, y)
    return X_res, y_res


# PIPELINE FUNCTIONS
def aggregate_data(observation_file: str, meta_file: str) -> pd.DataFrame:
    """This method aggregates the original observations with the collected metadata to form a single cohesive dataframe

    Args:
        observation_file (str): The file name, that points to the file containing the processed iNaturalist observations
        meta_file (str): The file name, that points to the file containing the metadata for the processed iNaturalist observations
    """
    obs_df = pd.read_csv(root_path + data_path + observation_file, index_col=0)  # Read in the csv files
    meta_df = pd.read_csv(root_path + data_path + meta_file, index_col=0)

    obs_df = obs_df.drop(columns=['observed_on', 'local_time_observed_at', 'positional_accuracy'])  # Drop repeated/ non-essential columns
    meta_df = meta_df.drop(columns=['lat', 'long', 'time'])

    df = pd.merge(obs_df, meta_df, how='inner', left_index=True, right_index=True)  # Merge the two dataframes
    return df


def nn_binary_label_handling(y):
    """Method handles the OHE of a binary case to ensure that OHE values returned are of the form [1, 0] or [0, 1].

    Args:
        y (Series): The labels in the form of either 1 or 0 to be transformed into a binary OHE

    Returns:
        (Series): Returns a Series containing OHE labels of the form [1, 0] or [0, 1]
    """
    return np.hstack((1 - y.reshape(-1, 1), y.reshape(-1, 1)))


def sub_species_detection(x):
    """Method uses the scientific name of observations to extract the subspecies name when there are more than
    three words present (3 names describe a subspecies)

    Args:
        x (DataFrame row): This variable represents a dataframe row containing the 'scientific_name' column.

    Returns:
        (DataFrame row): The method returns the dataframe row with an additional column value 'sub_species' if it could be extracted from the scientific name.
    """
    name_count = len(x['scientific_name'].split())  # Determine the number of names in the scientific name
    x['sub_species'] = np.nan  # Initialize the subspecies value
    if name_count >= 3:
        x['sub_species'] = x['scientific_name']
    return x


def ohe_month(df: pd.DataFrame):
    """Method performs OHE on the month feature of each observation

    Args:
        df (DataFrame): The dataframe containing all observation data from the processed data directory.
    """
    cats = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # Initialise the possible categories
    cat_type = CategoricalDtype(categories=cats)

    df['season'] = df['season'].astype(cat_type)

    df = pd.get_dummies(df, prefix='szn', columns=['season'], drop_first=True)  # Perform OHE
    return df


def land_mask(x):
    """This method determines if the observation coordinates are terrestrial or aquatic in nature.

    This method uses the Globe library to evaluate the location against a land mask.
    If the observation is terrestrial a value of 1 is given. If not 0 is given.

    Args:
        x (DataFrame row): This variable represents a dataframe row containing the 'latitude' and 'longitude' columns.

    Returns:
        (DataFrame row): The method returns the dataframe row with an additional binary column value 'land'
    """
    latitude = x['latitude']
    longitude = x['longitude']
    x['land'] = int(globe.is_land(latitude, longitude))
    return x


def elevation_clean(x):  # If observation is terrestrial, 0.0m elevation requires modification
    """This method performs a logical check on each observation's elevation based on the land feature value.

    The Open-Meteo API sets elevation to be 0m if the elevation is unknown.
    If the elevation is 0m and the land value is 1 (indicating a terrestrial sighting), then the elevation is set to NaN value.
    This NaN value will be modified within the pipeline with the species average elevation

    Args:
        x (DataFrame row): This variable represents a dataframe row containing the 'land' column.

    Returns:
        (DataFrame row): The method returns the dataframe row with the 'elevation' feature adjusted.
    """
    land = x['land']
    elevation = x['elevation']

    if land == 1 and elevation == 0:
        x['elevation'] = np.nan
    return x


def localize_sunrise_sunset(x):
    """This method localizes the sunrise and sunset times based on the time zone to aid in the light/ dark feature.
    Args:
        x (DataFrame row): This variable represents a dataframe row containing the 'sunrise' and 'sunset' columns.

    Returns:
        (DataFrame row): The method returns the dataframe row with the 'sunrise' and 'sunset' features adjusted.
    """
    timezone = pytz.timezone(x['time_zone'])
    x['sunrise'] = x['sunrise'].replace(tzinfo=timezone)
    x['sunset'] = x['sunset'].replace(tzinfo=timezone)
    return x


def dark_light_calc(x):
    """This method performs the dark/ light feature creation based on the time of observation and the sunrise & sunset times
    Args:
        x (DataFrame row): This variable represents a dataframe row.

    Returns:
        (DataFrame row): The method returns the dataframe row with a new binary 'light' column
    """
    timezone = pytz.timezone(x['time_zone'])  # Extract the timezone
    sunrise_utc = x['sunrise']
    sunset_utc = x['sunset']

    observ_time = x['observed_on'].replace(tzinfo=pytz.utc)  # Place time zone info with utc (this is required to localize the timezone in the next step)
    observ_time = x['observed_on'].astimezone(timezone)  # Generate the observed timezone in local time

    x['light'] = int(sunrise_utc <= observ_time <= sunset_utc)  # Logical operators cast into integer form create the binary light (1) or dark (0) value
    return x


def day_night_calculation(df: pd.DataFrame):
    """This method provides the overall process to create the light/ dark feature.

    This method converts the time of observation, sunrise, and sunset into local times.
    Local times are compared to determine light or dark.
    The sunrise and sunset columns are removed as they are no longer required.

    Args:
        df (DataFrame): The dataframe containing all observation data from the processed data directory.

    Returns:
        (DataFrame): The dataframe with the additional light column, and the sunrise & sunset columns removed.
    """
    # Convert to datetime objects. Remove NaT values from resulting transformation
    df['sunrise'] = pd.to_datetime(df['sunrise'],
                                   format="%Y-%m-%dT%H:%M",
                                   errors='coerce')
    df['sunset'] = pd.to_datetime(df['sunset'],
                                  format="%Y-%m-%dT%H:%M",
                                  errors='coerce')
    df = df.dropna(subset=['sunrise', 'sunset'])

    df = df.apply(lambda x: localize_sunrise_sunset(x), axis=1)  # Localize sunrise and sunset times to be timezone aware

    df = df.apply(lambda x: dark_light_calc(x), axis=1)  # Dark/ light calc based on sunrise and sunset times

    df = df.drop(columns=['sunrise', 'sunset'])

    return df


def season_calc(x):
    """This method determines the season in which an observation occurred based on the month of observation

    Args:
        x (DataFrame row): This variable represents a dataframe row.

    Returns:
        (DataFrame row): The method returns the dataframe row with a new season feature
    """
    hemisphere = x['hemisphere']
    month = x['month']
    season = 0

    if hemisphere == 1:  # Northern hemisphere
        winter, spring, summer, autumn = [12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]  # Order of months starting with winter season
        seasons = [winter, spring, summer, autumn]
        season = north_south_calc(month, seasons)
    else:  # Southern hemisphere
        winter, spring, summer, autumn = [6, 7, 8], [9, 10, 11], [12, 1, 2], [3, 4, 5]  # Order of months starting with winter seasons
        seasons = [winter, spring, summer, autumn]
        season = north_south_calc(month, seasons)

    x['season'] = season
    return x


def north_south_calc(month: int, seasons: list):
    """This method determined the current season when provided with a month and a list of seasonal months.

     Note, this method is used within the `season_calc()` method.

     Args:
         month (int): The integer value of the month of sighting [1-12]
         seasons (list): The list of months seperated by season, starting with winter. Example of northern hemisphere [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]

    Returns:
        (str): The season categorical variable
     """
    seasons_dict = {0: 'Winter', 1: 'Spring', 2: 'Summer', 3: 'Autumn'}
    season_id = 0
    for i in range(len(seasons)):
        if month in seasons[i]:
            season_id = i
    return seasons_dict[season_id]


def ohe_season(df):
    """This method OHE the season categorical feature

    Args:
        df (DataFrame): The dataframe containing all observation data from the processed data directory.

    Returns:
        (DataFrame): The dataframe with the season feature OHE (this results in additional columns within the dataframe)
    """
    cats = ['Winter', 'Spring', 'Summer', 'Autumn']  # Season categorical variables to expect
    cat_type = CategoricalDtype(categories=cats)

    df['season'] = df['season'].astype(cat_type)

    df = pd.get_dummies(df,
                        prefix='szn',
                        columns=['season'],
                        drop_first=True)
    return df
