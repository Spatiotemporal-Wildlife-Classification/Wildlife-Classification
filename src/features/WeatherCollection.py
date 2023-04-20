import requests

import csv

from src.structure import Config

root_path = Config.root_dir()
"""string: The root file path of the project"""
data_path = '/data/processed/'
"""The data path to the directory of processed data."""
dsn_endpoint = "http://139.144.179.74:5000/"


def retrieve_completed_jobs(start: int = None, end: int = None):
    retrieve_endpoint = "jobs_completed/"
    params = {"start": start, "end": end}
    try:
        req = requests.get(url=dsn_endpoint + retrieve_endpoint, params=params, timeout=5)
        return req.json()
    except Exception:
        return None


def write_to_csv(data):
    file = open(root_path + data_path + '/felids_meta.csv', 'a')
    for obs in data:
        obs = obs.replace('\n', '')
        obs = obs.split(',')
        writer = csv.writer(file)
        writer.writerow(obs)


def retrieve_weather_data(start: int = None, end: int = None):
    data = retrieve_completed_jobs(start, end)
    write_to_csv(data)


def get_error_counts():
    error_endpoint = "total_errors/"

    try:
        req = requests.get(url=dsn_endpoint + error_endpoint, timeout=5)
        return req.json()
    except Exception:
        print('Error retrieving error counts')


if __name__ == "__main__":
    retrieve_weather_data()
    error_counts = get_error_counts()
    print(error_counts)
