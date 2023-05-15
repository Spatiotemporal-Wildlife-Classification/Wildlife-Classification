from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from imblearn.over_sampling import RandomOverSampler


from src.structure.Config import root_dir

# General
import numpy as np
import pytz
from pandas.api.types import CategoricalDtype
import pandas as pd

# Geolocation libraries
from global_land_mask import globe


root_path = root_dir()
data_path = '/data/processed/'
data_destination = '/notebooks/model_comparison_cache/'

# K-means information
k_max = 100
k_interval = 2
k_init = 4


def aggregate_data(observation_file: str, meta_file: str) -> pd.DataFrame:
    obs_df = pd.read_csv(root_path + data_path + observation_file, index_col=0)
    meta_df = pd.read_csv(root_path + data_path + meta_file, index_col=0)

    obs_df = obs_df.drop(columns=['observed_on', 'local_time_observed_at', 'positional_accuracy'])
    meta_df = meta_df.drop(columns=['lat', 'long', 'time'])

    df = pd.merge(obs_df, meta_df, how='inner', left_index=True, right_index=True)
    return df


def decision_tree_data(df: pd.DataFrame, taxon_target: str, k_cluster, validation_file: str):
    k_means = train_kmeans(df)
    X, y = tree_pipeline(df, k_means, taxon_target, validation_file)
    return X, y


def xgb_data(df: pd.DataFrame, taxon_target: str, k_cluster, validation_file: str):
    k_means = train_kmeans(df)
    X, y = xgb_pipeline(df, k_means, taxon_target, validation_file)
    return X, y


def neural_network_data(df: pd.DataFrame, taxon_target: str, k_cluster, validation_file: str):
    k_means = train_kmeans(df)
    X, y, lb, classes = nn_pipeline(df, k_means, taxon_target, validation_file)
    return X, y, lb, classes


## DECISION TREE PIPELINE ##
def tree_pipeline(df, k_means, taxon_target, validation_file: str):
    ## CLEAN UP##

    # Remove non-essential columns
    df = df.drop(columns=['geoprivacy', 'taxon_geoprivacy', 'taxon_id', 'license', 'image_url'])

    # Remove null species names
    df = df.dropna(subset=['taxon_species_name'])

    # Drop null positional accuracies
    df = df.dropna(subset=['public_positional_accuracy'])

    # Positional Accuracy Restriction
    df = df[df['public_positional_accuracy'] <= 40000]
    df = df.drop(columns=['public_positional_accuracy'])

    ## TRANSFORM ##

    # Generate sub-species and drop scientific name
    df = df.apply(lambda x: sub_species_detection(x), axis=1)
    df = df.drop(columns=['scientific_name'])

    # Remove species with less than 5 observations
    df = df[df.groupby(taxon_target).common_name.transform('count') >= 10].copy()

    # Location Centroid Feature
    df['location_cluster'] = k_means.predict(df[['latitude', 'longitude']])

    # Terrestrial vs Land Feature
    df['land'] = 1

    # Elevation Logical Path, dependent on land
    df = df.apply(lambda x: elevation_clean(x), axis=1)
    df['elevation'] = df['elevation'].fillna(df.groupby('taxon_species_name')['elevation'].transform('mean'))

    # Northern and Southern Hemisphere OHE
    df['hemisphere'] = (df['latitude'] >= 0).astype(int)
    df = df.drop(columns=['latitude', 'longitude'])

    # Datetime Transformation
    df['observed_on'] = pd.to_datetime(df['observed_on'],
                                       format="%Y-%m-%d %H:%M:%S%z",
                                       utc=True)

    # Month Feature
    df['month'] = df['observed_on'].dt.month

    # Hour Feature
    df['hour'] = df.apply(lambda x: x['observed_on'].astimezone(pytz.timezone(x['time_zone'])).hour, axis=1)

    # Day/Night Feature
    df = day_night_calculation(df)

    # Season Feature
    df = df.apply(lambda x: season_calc(x), axis=1)
    df = ohe_season(df)

    # Drop observed on column as date & time transformations are complete
    df = df.drop(columns=['observed_on', 'time_zone'])

    # Create validation set for further testing
    df = validation_set(df, taxon_target, validation_file)

    ## TRAIN & TEST DATA
    # Retrieve labels
    taxon_y = df[taxon_target]

    # Sub-specie contains null values, if selected as target taxon. Remove
    if taxon_y.isnull().any():
        df = df.dropna(subset=[taxon_target])

    y = df[taxon_target]
    X = df.drop(columns=['taxon_kingdom_name', 'taxon_phylum_name',
                         'taxon_class_name', 'taxon_order_name', 'taxon_family_name',
                         'taxon_genus_name', 'taxon_species_name', 'sub_species', 'common_name'])

    # Resample dataset to reduce imbalance
    X, y = over_sample(X, y)

    return X, y


## XGBOOST PIPELINE ##
def xgb_pipeline(df, k_means, taxon_target, validation_file: str):
    ## CLEAN UP##

    # Remove non-essential columns
    df = df.drop(columns=['geoprivacy', 'taxon_geoprivacy', 'taxon_id', 'license', 'image_url'])

    # Remove null species names
    df = df.dropna(subset=['taxon_species_name'])

    # Drop null positional accuracies
    df = df.dropna(subset=['public_positional_accuracy'])

    # Positional Accuracy Restriction
    df = df[df['public_positional_accuracy'] <= 40000]
    df = df.drop(columns=['public_positional_accuracy'])

    ## TRANSFORM ##

    # Generate sub-species and rop scientific name
    df = df.apply(lambda x: sub_species_detection(x), axis=1)
    df = df.drop(columns=['scientific_name'])

    # Remove species with less than 5 observations
    df = df[df.groupby(taxon_target).common_name.transform('count') >= 10].copy()

    # Location Centroid Feature
    df['location_cluster'] = k_means.predict(df[['latitude', 'longitude']])

    # Terrestrial vs Land Feature
    df['land'] = 1

    # Elevation Logical Path, dependent on land
    df = df.apply(lambda x: elevation_clean(x), axis=1)
    df['elevation'] = df['elevation'].fillna(df.groupby('taxon_species_name')['elevation'].transform('mean'))

    # Northern and Southern Hemisphere OHE
    df['hemisphere'] = (df['latitude'] >= 0).astype(int)
    df = df.drop(columns=['latitude', 'longitude'])

    # Datetime Transformation
    df['observed_on'] = pd.to_datetime(df['observed_on'],
                                       format="%Y-%m-%d %H:%M:%S%z",
                                       utc=True)

    # Month Feature
    df['month'] = df['observed_on'].dt.month

    # Hour Feature
    df['hour'] = df.apply(lambda x: x['observed_on'].astimezone(pytz.timezone(x['time_zone'])).hour, axis=1)

    # Day/Night Feature
    df = day_night_calculation(df)

    # Season Feature
    df = df.apply(lambda x: season_calc(x), axis=1)
    df = ohe_season(df)

    # Drop observed on column as date & time transformations are complete
    df = df.drop(columns=['observed_on', 'time_zone'])

    # Create validation set for further testing
    df = validation_set(df, taxon_target, validation_file)

    ## TRAIN & TEST DATA
    # Retrieve labels
    taxon_y = df[taxon_target]

    # Sub-specie contains null values, if selected as target taxon. Remove
    if taxon_y.isnull().any():
        df = df.dropna(subset=[taxon_target])

    y = df[taxon_target]
    X = df.drop(columns=['taxon_kingdom_name', 'taxon_phylum_name',
                         'taxon_class_name', 'taxon_order_name', 'taxon_family_name',
                         'taxon_genus_name', 'taxon_species_name', 'sub_species', 'common_name'])

    # Resample dataset to reduce imbalance
    X, y = over_sample(X, y)

    # Encode labels
    classes = y.nunique()
    lb = LabelBinarizer()
    lb.fit(y)
    y = lb.transform(y)

    # Binary check
    if classes == 2:
        y = nn_binary_label_handling(y)

    return X, y


## NEURAL NETWORK PIPELINE
def nn_pipeline(df, k_means, taxon_target, validation_file: str):
    ## CLEAN UP##

    # Remove non-essential columns
    df = df.drop(columns=['geoprivacy', 'taxon_geoprivacy', 'taxon_id', 'license', 'image_url',
                            'weathercode_daily', 'weathercode_hourly'])

    # Drop null positional accuracies
    df = df.dropna(subset=['public_positional_accuracy'])

    # Positional Accuracy Restriction
    df = df[df['public_positional_accuracy'] <= 40000]
    df = df.drop(columns=['public_positional_accuracy'])

    ## TRANSFORM ##

    # Generate sub-species and rop scientific name
    df = df.apply(lambda x: sub_species_detection(x), axis=1)
    df = df.drop(columns=['scientific_name'])

    # Remove species with less than 10 observations
    df = df[df.groupby(taxon_target).common_name.transform('count') >= 10].copy()

    # Location Centroid Feature
    df['location_cluster'] = k_means.predict(df[['latitude', 'longitude']])
    df = pd.get_dummies(df,
                        prefix='loc',
                        columns=['location_cluster'],
                        drop_first=True)

    # Terrestrial vs Land Feature
    df['land'] = 1

    # Elevation Logical Path, dependent on land
    df = df.apply(lambda x: elevation_clean(x), axis=1)
    df['elevation'] = df['elevation'].fillna(df.groupby('taxon_species_name')['elevation'].transform('mean'))

    # Northern and Southern Hemisphere OHE
    df['hemisphere'] = (df['latitude'] >= 0).astype(int)
    df = df.drop(columns=['latitude', 'longitude'])

    # Datetime Transformation
    df['observed_on'] = pd.to_datetime(df['observed_on'],
                                           format="%Y-%m-%d %H:%M:%S%z",
                                           utc=True)

    # Month Feature
    df['month'] = df['observed_on'].dt.month

    # Hour Feature
    df['hour'] = df.apply(lambda x: x['observed_on'].astimezone(pytz.timezone(x['time_zone'])).hour, axis=1)
    df = pd.get_dummies(df,
                        prefix='hr',
                        columns=['hour'],
                        drop_first=True)

    # Day/Night Feature
    df = day_night_calculation(df)

    # Season Feature
    df = df.apply(lambda x: season_calc(x), axis=1)
    df = ohe_season(df)

    # Season is dependent on month, hence month ohe here
    df = pd.get_dummies(df,
                        prefix='mnth',
                        columns=['month'],
                        drop_first=True)

    # Drop observed on column as date & time transformations are complete
    df = df.drop(columns=['observed_on', 'time_zone'])

    # Create validation set for further testing
    df = validation_set(df, taxon_target, validation_file)


    ## TRAIN & TEST DATA
    # Retrieve labels
    taxon_y = df[taxon_target]

    # Sub-specie contains null values, if selected as target taxon. Remove
    if taxon_y.isnull().any():
        df = df.dropna(subset=[taxon_target])

    y = df[taxon_target]
    X = df.drop(columns=['taxon_kingdom_name', 'taxon_phylum_name',
                             'taxon_class_name', 'taxon_order_name', 'taxon_family_name',
                             'taxon_genus_name', 'taxon_species_name', 'sub_species', 'common_name'])

    X, y = over_sample(X, y)


    # Encode labels
    classes = y.nunique()
    lb = LabelBinarizer()
    lb.fit(y)
    y = lb.transform(y)

    # Binary check
    if classes == 2:
        y = nn_binary_label_handling(y)


    # Min-max normalize data
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
    return X, y, lb, classes


## KMEANS ##
def train_kmeans(df: pd.DataFrame):
    location_df = df[['latitude', 'longitude']]

    sil_scores = calculate_optimal_k(location_df)
    k_values = range(k_init, k_max + k_interval, k_interval)

    silhouette_optimal = np.argmax(sil_scores)
    k_value = (k_values[silhouette_optimal])
    print("K-value: ", k_value)

    k_means = KMeans(n_clusters=k_value, n_init=10)
    k_means.fit(location_df)
    return k_means


# Use the Silhouette score to determine an optimal k
def calculate_optimal_k(data):
    sil = []

    k = k_init
    while k <= k_max and k <= len(data):
        k_means = KMeans(n_clusters=k, n_init=10).fit(data)
        labels = k_means.labels_

        sil.append(silhouette_score(data, labels))

        k += k_interval
    return sil


## VALIDATION SET ##
def validation_set(df: pd.DataFrame, target_taxon: str, file_name: str):
    # Ensure at least 4 of each species are present in evaluation dataset
    grouped = df.groupby([target_taxon]).sample(n=10, random_state=2)
    # Save evaluation dataset
    grouped.to_csv(root_path + data_destination + file_name)

    # Remove evaluation observations from df
    df = df.drop(grouped.index)
    return df


## OVER SAMPLING ##
def over_sample(X, y):
    ros = RandomOverSampler(sampling_strategy='minority',
                            random_state=2)
    X_res, y_res = ros.fit_resample(X, y)
    return X_res, y_res


## PIPELINE FUNCTIONS ##

def nn_binary_label_handling(y):
    return np.hstack((y, 1 - y))


def sub_species_detection(x):
    name_count = len(x['scientific_name'].split())
    x['sub_species'] = np.nan
    if name_count >= 3:
        x['sub_species'] = x['scientific_name']
    return x


def ohe_month(df):
    cats = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    cat_type = CategoricalDtype(categories=cats)

    df['season'] = df['season'].astype(cat_type)

    df = pd.get_dummies(df,
                        prefix='szn',
                        columns=['season'],
                        drop_first=True)
    return df


def land_mask(x):
    latitude = x['latitude']
    longitude = x['longitude']
    x['land'] = int(globe.is_land(latitude, longitude))
    return x


def elevation_clean(x):  # If observation is terrestrial, 0.0m elevation requires modification
    land = x['land']
    elevation = x['elevation']
    if land == 1 and elevation == 0:
        x['elevation'] = np.nan
    return x


def localize_sunrise_sunset(x):
    timezone = pytz.timezone(x['time_zone'])
    x['sunrise'] = x['sunrise'].replace(tzinfo=timezone)
    x['sunset'] = x['sunset'].replace(tzinfo=timezone)
    return x


def dark_light_calc(x):
    timezone = pytz.timezone(x['time_zone'])
    sunrise_utc = x['sunrise']
    sunset_utc = x['sunset']

    observ_time = x['observed_on'].replace(tzinfo=pytz.utc)
    observ_time = x['observed_on'].astimezone(timezone)

    x['light'] = int(sunrise_utc <= observ_time <= sunset_utc)
    return x


def day_night_calculation(df):
    # Convert to datetime objects. Remove NaT values from resulting transformation
    df['sunrise'] = pd.to_datetime(df['sunrise'],
                                   format="%Y-%m-%dT%H:%M",
                                   errors='coerce')
    df['sunset'] = pd.to_datetime(df['sunset'],
                                  format="%Y-%m-%dT%H:%M",
                                  errors='coerce')
    df = df.dropna(subset=['sunrise', 'sunset'])

    # Localize sunrise and sunset times to be timezone aware
    df = df.apply(lambda x: localize_sunrise_sunset(x), axis=1)

    # Dark/ light calc based on sunrise and sunset times
    df = df.apply(lambda x: dark_light_calc(x), axis=1)

    df = df.drop(columns=['sunrise', 'sunset'])

    return df


def season_calc(x):
    hemisphere = x['hemisphere']
    month = x['month']
    season = 0
    if hemisphere == 1:  # Northern hemisphere
        winter, spring, summer, autumn = [12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]
        seasons = [winter, spring, summer, autumn]
        season = north_south_calc(month, seasons)
    else:
        winter, spring, summer, autumn = [6, 7, 8], [9, 10, 11], [12, 1, 2], [3, 4, 5]
        seasons = [winter, spring, summer, autumn]
        season = north_south_calc(month, seasons)

    x['season'] = season
    return x


def north_south_calc(month: int, seasons: list):
    seasons_dict = {0: 'Winter', 1: 'Spring', 2: 'Summer', 3: 'Autumn'}
    season_id = 0
    for i in range(len(seasons)):
        if month in seasons[i]:
            season_id = i
    return seasons_dict[season_id]


def ohe_season(df):
    cats = ['Winter', 'Spring', 'Summer', 'Autumn']
    cat_type = CategoricalDtype(categories=cats)

    df['season'] = df['season'].astype(cat_type)

    df = pd.get_dummies(df,
                        prefix='szn',
                        columns=['season'],
                        drop_first=True)
    return df