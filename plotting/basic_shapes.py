"""
Construction of basic matplotlib patches.

These are mostly wrappers for the constructors of primitive patches,
allowing input arguments that I find more useful, along with a few
new constructors for shapes not in matplotlib.patches's defaults.

To eliminate the need for separate imports of matplotlib.pathces and
this module, some wrappers are included that exactly preserve the
original matplotlib.patches syntax (e.g., Circle).
"""


import numpy as np
import matplotlib.patches as patch


def square(center, side_length):
    """ Create a square with the given center and side length. """
    center = np.asarray(center)
    side_length = float(side_length)
    lower_left = center - 0.5*side_length
    return patch.Rectangle(lower_left, side_length, side_length)


def circle(center, radius):
    """ Create a circle with the given center and radius. """
    return patch.Circle(center, radius)