""" This file serves to download the weather data collected by the Distributed Scraping Network (DSN)

    Attributes:
        root_path (str): The absolute path to the root of the project directory
        data_path (string): The path to the location of the weather data storage location
        dsn_endpoint (str): The endpoint of the DSN API
"""

import requests

import csv

from src.structure import Config

# Paths
root_path = Config.root_dir()
data_path = '/data/obs_and_meta/processed/'
dsn_endpoint = "http://139.144.179.74:5000/"


def retrieve_completed_jobs(start: int = None, end: int = None):
    """Method to perform GET request retrieving the stored weather data at the DSN server.

    The start and end parameters provide the index range of the jobs to be returned.
    For example, specifying start = 0, and end = 1000 will return the first 1000 jobs completed and stored on the DSN
    Args:
        start (int): Specification of the start index where collection should start from (included)
        end (int): Specification of the end index where collection should end (included)

    Returns:
        (Json): The Json of the request containing all of the requested weather data. If the request fails, None is returned.
    """
    retrieve_endpoint = "jobs_completed/"  # Specify endpoint
    params = {"start": start, "end": end}  # GET request parameters
    try:
        req = requests.get(url=dsn_endpoint + retrieve_endpoint, params=params, timeout=5)
        return req.json()
    except Exception:
        return None


def write_to_csv(data, file):
    """Method to write the collected weather data to file

    Args:
        data (Json): The weather data collected in Json form
        file (str): The file name where the data should be written to
    """
    file = open(root_path + data_path + file, 'a')
    for obs in data:
        obs = obs.replace('\n', '')  # Remove all newline characters
        obs = obs.split(',')  # Split at comma delimiter

        writer = csv.writer(file)
        writer.writerow(obs)


def retrieve_weather_data(start: int = None, end: int = None, file_name=''):
    """Method specifies the process of collection followed by writing the collected data into a single method.
    Args:
        start (int): Specification of the start index where collection should start from (included)
        end (int): Specification of the end index where collection should end (included)
        file_name (str): A string specifying the name of the file to write the weather data to (must be a csv file)
    """
    data = retrieve_completed_jobs(start, end)
    write_to_csv(data, file_name)


def get_error_counts():
    """The method performs a GET request to the DSN to get the total error count.

    The error count represents the number of jobs which failed to collect weather data for a variety of reasons.

    Returns:
        (Json): The error count in Json format"""
    error_endpoint = "total_errors/"  # DSN error count endpoint

    try:
        req = requests.get(url=dsn_endpoint + error_endpoint, timeout=5)
        return req.json()
    except Exception:
        print('Error retrieving error counts')


if __name__ == "__main__":
    retrieve_weather_data(file_name='/felids_meta.csv')  # Weather collection process
    error_counts = get_error_counts()  # Total error count
    print(error_counts)  # Display total errors
