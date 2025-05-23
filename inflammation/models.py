"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row
contains inflammation data for a single patient taken over a number of days
and each column represents a single day across all patients.
"""

import numpy as np
import json


def load_json(filename):
    """Load a Numpy array from a JSON

    :param filename: Filename of JSON to load
    """
    with open(filename, 'r', encoding='utf-8') as file:
        data_as_json = json.load(file)
        return [np.array(entry['observations']) for entry in data_as_json]


def load_csv(filename):
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    """
    return np.loadtxt(fname=filename, delimiter=',')


def daily_mean(data):
    """
    Calculate the daily mean of a 2D inflammation data array.

    :param data: A 2D array with inflammation data
    :returns: An array of mean values of measurements for each day
    """
    return np.mean(data, axis=0)


def daily_max(data):
    """
    Calculate the daily max of a 2D inflammation data array.

    :param data: A 2D array with inflammation data
    :returns: An array of max values of measurements for each day
    """
    return np.max(data, axis=0)


def daily_min(data):
    """
    Calculate the daily min of a 2D inflammation data array.

    :param data: A 2D array with inflammation data
    :returns: An array of min values of measurements for each day
    """
    return np.min(data, axis=0)


def patient_normalise(data):
    """
    Normalise patient data from an 2D inflammation data array.

    :param data: A 2D array with inflammation data
    """
    if np.any(data < 0):
        raise ValueError("Cannot normalise data with negative values")
    max_array = np.nanmax(data, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalised = data / max_array[:, np.newaxis]
    normalised[np.isnan(normalised)] = 0
    normalised[normalised < 0] = 0
    return normalised
