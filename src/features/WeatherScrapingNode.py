import sys
import traceback
from time import sleep

import requests
from datetime import datetime
import json

dsn_endpoint = "http://139.144.179.74:5000/"

weather_endpoint = "https://archive-api.open-meteo.com/v1/archive?"

hourly_weather_var = ["temperature_2m", "relativehumidity_2m", "dewpoint_2m", "apparent_temperature",
                      "surface_pressure", "precipitation", "rain", "snowfall", "cloudcover",
                      "cloudcover_low", "cloudcover_mid", "cloudcover_high", "shortwave_radiation",
                      "direct_radiation", "diffuse_radiation", "windspeed_10m", "windspeed_100m",
                      "winddirection_10m", "winddirection_100m", "windgusts_10m", "et0_fao_evapotranspiration",
                      "weathercode", "vapor_pressure_deficit", "soil_temperature_0_to_7cm",
                      "soil_temperature_7_to_28cm",
                      "soil_temperature_28_to_100cm", "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm",
                      "soil_moisture_28_to_100cm"]

daily_weather_var = ["weathercode", "temperature_2m_max", "temperature_2m_min", "apparent_temperature_max",
                     "apparent_temperature_min", "precipitation_sum", "rain_sum", "snowfall_sum",
                     "precipitation_hours", "sunrise", "sunset", "windspeed_10m_max", "windgusts_10m_max",
                     "winddirection_10m_dominant", "shortwave_radiation_sum", "et0_fao_evapotranspiration"]

job_limit = 10000

rate_limit = 1.0


def get_job_info():
    job_info_endpoint = "job/"
    try:
        request = requests.get(url=dsn_endpoint + job_info_endpoint, timeout=5)
        return request.json()
    except Exception as e:
        traceback.print_exc()
        return None


def date_check(job):
    observed_on = job['obs_time']
    observed_on = datetime.strptime(observed_on, "%Y-%m-%d %H:%M:%S%z")  # Convert obs_on into datetime

    start_date = job['start_date']
    start_date = datetime.strptime(start_date, "%Y-%m-%d")  # Convert start_date into datetime

    if start_date.day != observed_on.day:
        job['start_date'] = observed_on.strftime("%Y-%m-%d")
        job['end_date'] = observed_on.strftime("%Y-%m-%d")

    return job


def execute_request(job):
    """Method performs weather request to Open_meteo
    The auto timezone parameters means that coordinates will be automatically resolved to local timezone
    """
    global rate_limit

    sleep(rate_limit)
    params = {"latitude": job["latitude"],
              "longitude": job["longitude"],
              "start_date": job["start_date"],
              "end_date": job["end_date"],
              "timezone": "auto",
              "hourly": hourly_weather_var,
              "daily": daily_weather_var}
    try:
        req = requests.get(url=weather_endpoint, params=params, timeout=5)
        if req.status_code == 403:  # Too many request response.
            rate_limit = rate_limit + (rate_limit * 0.2)
        return req.json()
    except Exception:
        print("Error in Ope-Meteo request")
        return None


def job_complete_response(weather_data, job):
    completion_endpoint = "job/"  # Specify completion endpoint
    try:
        weather_hour, hour_index = determine_hour(weather_data, job["obs_time"])  # Get hour of observation
        data = job_complete_formatting(weather_data, job, weather_hour, hour_index)  # Format collected data
    except Exception:
        print("Error in data formatting")

    try:
        req = requests.post(url=dsn_endpoint + completion_endpoint, data=json.dumps(data), timeout=5)  # Post request
    except Exception:
        print("Error in job completion transfer")


def job_complete_formatting(weather_data, job, weather_hour, hour_index):
    data = {"id": job["id"],
            "lat": job["latitude"],
            "long": job["longitude"],
            "observed_on": job["obs_time"],
            "time_zone": weather_data["timezone"],
            "elevation": weather_data["elevation"],
            "time": weather_hour,

            # Hourly variables
            "temperature_2m": float(weather_data["hourly"]["temperature_2m"][hour_index]),
            "relativehumidity_2m": float(weather_data["hourly"]["relativehumidity_2m"][hour_index]),
            "dewpoint_2m": float(weather_data["hourly"]["dewpoint_2m"][hour_index]),
            "apparent_temperature": float(weather_data["hourly"]["apparent_temperature"][hour_index]),
            "surface_pressure": float(weather_data["hourly"]["surface_pressure"][hour_index]),
            "precipitation": float(weather_data["hourly"]["precipitation"][hour_index]),
            "rain": float(weather_data["hourly"]["rain"][hour_index]),
            "snowfall": float(weather_data["hourly"]["snowfall"][hour_index]),
            "cloudcover": float(weather_data["hourly"]["cloudcover"][hour_index]),
            "cloudcover_low": float(weather_data["hourly"]["cloudcover_low"][hour_index]),
            "cloudcover_mid": float(weather_data["hourly"]["cloudcover_mid"][hour_index]),
            "cloudcover_high": float(weather_data["hourly"]["cloudcover_high"][hour_index]),
            "shortwave_radiation": float(weather_data["hourly"]["shortwave_radiation"][hour_index]),
            "direct_radiation": float(weather_data["hourly"]["direct_radiation"][hour_index]),
            "diffuse_radiation": float(weather_data["hourly"]["diffuse_radiation"][hour_index]),
            "windspeed_10m": float(weather_data["hourly"]["windspeed_10m"][hour_index]),
            "windspeed_100m": float(weather_data["hourly"]["windspeed_100m"][hour_index]),
            "winddirection_10m": float(weather_data["hourly"]["winddirection_10m"][hour_index]),
            "winddirection_100m": float(weather_data["hourly"]["winddirection_100m"][hour_index]),
            "windgusts_10m": float(weather_data["hourly"]["windgusts_10m"][hour_index]),
            "et0_fao_evapotranspiration_hourly": float(
                weather_data["hourly"]["et0_fao_evapotranspiration"][hour_index]),
            "weathercode_hourly": int(weather_data["hourly"]["weathercode"][hour_index]),
            "vapor_pressure_deficit": float(weather_data["hourly"]["vapor_pressure_deficit"][hour_index]),
            "soil_temperature_0_to_7cm": float(weather_data["hourly"]["soil_temperature_0_to_7cm"][hour_index]),
            "soil_temperature_7_to_28cm": float(weather_data["hourly"]["soil_temperature_7_to_28cm"][hour_index]),
            "soil_temperature_28_to_100cm": float(weather_data["hourly"]["soil_temperature_28_to_100cm"][hour_index]),
            "soil_moisture_0_to_7cm": float(weather_data["hourly"]["soil_moisture_0_to_7cm"][hour_index]),
            "soil_moisture_7_to_28cm": float(weather_data["hourly"]["soil_moisture_7_to_28cm"][hour_index]),
            "soil_moisture_28_to_100cm": float(weather_data["hourly"]["soil_moisture_28_to_100cm"][hour_index]),

            # Daily Variables
            "weathercode_daily": int(weather_data["daily"]["weathercode"][0]),
            "temperature_2m_max": float(weather_data["daily"]["temperature_2m_max"][0]),
            "temperature_2m_min": float(weather_data["daily"]["temperature_2m_min"][0]),
            "apparent_temperature_max": float(weather_data["daily"]["apparent_temperature_max"][0]),
            "apparent_temperature_min": float(weather_data["daily"]["apparent_temperature_min"][0]),
            "precipitation_sum": float(weather_data["daily"]["precipitation_sum"][0]),
            "rain_sum": float(weather_data["daily"]["rain_sum"][0]),
            "snowfall_sum": float(weather_data["daily"]["snowfall_sum"][0]),
            "precipitation_hours": float(weather_data["daily"]["precipitation_hours"][0]),
            "sunrise": str(weather_data["daily"]["sunrise"][0]),
            "sunset": str(weather_data["daily"]["sunset"][0]),
            "windspeed_10m_max": float(weather_data["daily"]["windspeed_10m_max"][0]),
            "windgusts_10m_max": float(weather_data["daily"]["windgusts_10m_max"][0]),
            "winddirection_10m_dominant": float(weather_data["daily"]["winddirection_10m_dominant"][0]),
            "shortwave_radiation_sum": float(weather_data["daily"]["shortwave_radiation_sum"][0]),
            "et0_fao_evapotranspiration_daily": float(weather_data["daily"]["et0_fao_evapotranspiration"][0]),
            }
    return data


def determine_hour(weather_data, observed_on):
    observed_on = datetime.strptime(observed_on, "%Y-%m-%d %H:%M:%S%z")  # Convert obs_on into datetime
    observed_on = observed_on.replace(microsecond=0, second=0, minute=0)  # Round down to nearest hour
    search_time = observed_on.strftime("%Y-%m-%dT%H:%M")  # Generate string matching request hours

    hourly_index = weather_data["hourly"]["time"].index(search_time)
    return search_time, hourly_index


def send_error_response():
    error_endpoint = "error/"

    try:
        requests.post(url=dsn_endpoint + error_endpoint, timeout=5)
    except Exception:
        print("Timeout updating error count")


def progress_bar(start_time: datetime, job_no: int):
    running_time = datetime.now() - start_time
    progress_bar_length = 50  # Specify progress bar length
    percentage_complete = float(job_no) / float(job_limit)
    filled = int(progress_bar_length * percentage_complete)  # Determine fill of percentage bar
    bar = '=' * filled + '-' * (progress_bar_length - filled)  # Modify bar with fill
    percentage_display = round(100 * percentage_complete, 3)  # Calculate percentage
    print(f"\r[{bar}] {percentage_display}%s\tcurrent observations: {job_no} / {job_limit} \t running time: {running_time}")


def scraping_node_process():
    start_time = datetime.now()
    for job_no in range(job_limit):  # Loop through number of requests
        progress_bar(start_time, job_no)
        job = get_job_info()
        job = date_check(job)
        if job is None:
            sys.exit()
        weather_data = execute_request(job)
        if weather_data is None:
            send_error_response()
            continue
        job_complete_response(weather_data, job)


if __name__ == "__main__":
    scraping_node_process()