""" Various plotting convenience functions """


import numpy as np
import matplotlib.pyplot as plt


def linear_symmetric_range(values, center_point):
    """
    Return an interval containing all elements of the passed values,
    and with a central value as given.  Interval is returned as a
    2-element, 1D array.
    """
    half_width = np.max(np.absolute(values - center_point))
    return np.array([center_point - half_width, center_point + half_width])