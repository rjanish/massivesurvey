"""Module for plotting a scalar field with spacial discretization."""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.collections as collect


def plot_scalar(axes, points, values, marker_func,
                color_limits=None, norm=None, colormap=cm.gray):
    """
    Plot a space-discretized, color-mapped scalar field to axes.

    Args:
    axes - matplotlib Axes
        Axes on which to plot the scalar field
    points - 2D arraylike, shape (N, 2)
        Cartesian coordinates of the N sampled field values
    values - 1D arraylike, shape N
        Values of the field at the N locations given in points.
    marker_func - func
        This is a function that returns the plotting marker for each
        sampled point. The syntax must be marker = func(coord, value),
        where coord is a 2-element coordinate array a given sample
        point, and value is its field value. marker is a matplotlib
        Patch that will be plotted on the location of coord. The
        color of marker does not need to be set by marker_func.
    norm - matplotlib Normalize, default is linear using data_limits
        The is a Normalize instance that will be used to map the
        passed values to colors. If not None, color_limits is ignored.
    color_limits - 2-element, 1D arraylike, default is values limits
        If norm is None, then a linear color mapping is used, with
        the limits of the linear region set by color_limits. By
        default the limits used will be the limits of values.
    colormap - matplotlib colormap, default=cm.gray
        The colormap used with norm to plot the color-valued field.

    Returns: all_patches
    all_patches - matplotlib PatchCollection
        This is the PatchCollection containing all of the markers
        plotted to axes, with colors set by values.
    """
    points = np.asarray(points, dtype=float)
    values = np.asarray(values, dtype=float)
    markers = [marker_func(pt, v) for pt, v in zip(points, values)]
    if (norm is None) and (color_limits is not None):
        norm = plt.Normalize(*color_limits)
    all_patches = collect.PatchCollection(markers, norm=norm, cmap=colormap)
    all_patches.set_array(values)  # sets color of each patch
    axes.add_collection(all_patches)
    return all_patches