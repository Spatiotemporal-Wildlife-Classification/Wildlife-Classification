import sys
from time import sleep

interval = 0
"""int: The interval in seconds to pause between consecutive API calls"""
request_limit = 10000
"""int: The total request to be executed during the collection process"""
percentage_increase = 0.05
"""float: The percentage to increase the interval if a 403 error encountered"""


def enforce_request_interval(request_no: int):
    """Method which enforces the API request interval by causing the program to sleep

    Args:
        request_no (int): The number of requests executed to the Open-Meteo endpoint
    """
    progress_bar(request_no)  # Update the progress bar
    sleep(interval)  # Perform sleep interval


def calculate_request_interval_batching(req_limit, duration):
    """Method calculates the interval time between requests. Updating the interval parameter

    Args:
        req_limit (int): The total number of requests to be executed
        duration (int): The duration over which the request must be executed (minutes)
    """
    global interval, request_limit
    request_limit = req_limit
    interval = 60 * (duration / request_limit)


def increase_interval():
    """Method increases the interval after encountering a 403 error."""
    global interval
    interval = interval + (percentage_increase * interval)


def progress_bar(request_no: int):
    """Method displays the progress bar of the API requests.

    Args:
        request_no (int): The current request number that is being executed. This serves as a status update.

    To utilize this method, use sys.stdout.flush() to render the result within a loop.
    Element unfortunately disconnected in order to display two status bars.
    """
    progress_bar_length = 100
    percentage_complete = (request_no / request_limit)
    filled = int(progress_bar_length * percentage_complete)
    running_time = (interval * request_no) / 60

    bar = '=' * filled + '-' * (progress_bar_length - filled)
    percentage_display = round(100 * percentage_complete, 1)
    sys.stdout.write('\n')
    sys.stdout.write('\r[%s] %s%s ... running: %s ... interval: %s' % (bar,
                                                                       percentage_display,
                                                                       '%',
                                                                       round(running_time, 2),
                                                                       round(interval, 2)))
