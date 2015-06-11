""" Miscellaneous geometry functions """


import numpy as np


def bounding_radius(polygon, center=None):
    """
    Return the radius of the smallest circle with the passed center
    that completely encloses the passed polygon.  If no center is
    given, the polygon's centroid is used (in this case, the returned
    result is the polygon's circumradius).  The input is assumed to
    be a shapely polygon object.
    """
    verticies = np.asarray(polygon.boundary.coords, dtype=float)
    if center is None:
        center = np.asarray(polygon.centroid.coords, dtype=float)
    sq_dists = np.sum((verticies - center)**2, axis=1)
    return np.sqrt(sq_dists.max())