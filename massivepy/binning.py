"""
This module contains functions for the spacial binning of IFU spectra,
and for the manipulation of sets of bins.
"""


import functools

import numpy as np
np.seterr(all='raise')   # force numpy warnings to raise exceptions
np.seterr(under='warn')  # for all but underflow warnings (numpy
                         # version 1.8.2 raises spurious underflows
                         # on some masked array computations)

import utilities as utl


def partition_quadparity(rad_interval, major_axis=None, aspect_ratio=None):
    """
    Partition an annulus into angular bins that have parity across
    both axes.

    Args:
    major_axis - float
        The major axis, as an angle counterclockwise from the
        x-axis, in radians. Bins will have parity across the major
        axis and the minor (major_axis + pi/2).
    aspect_ratio - float
        The target aspect ratio of the constructed bins, defined as
        angular_size/radial_size. The bins will have an aspect ratio
        no larger than the passed value, but it may be smaller.

    Returns: intervals
    intervals - 3D array
        The first axis delineates bins, the second axis list the
        disconnected polar boxes whose unions forms the bin, and the
        third gives the min and max angle of a particular polar box.
    """
    inner_radius, outer_radius = np.asarray(rad_interval, dtype=float)
    delta_r = outer_radius - inner_radius
    mid_r = 0.5*(outer_radius + inner_radius)
    target_bin_arclength = delta_r*aspect_ratio
    available_arclength = 0.5*np.pi*mid_r  # one quadrant
    num_in_quad = int(available_arclength/target_bin_arclength)
    if num_in_quad == 0:
        raise ValueError # invalid annulus - too thin for given aspect_ratio
    num_in_half = 2*num_in_quad
    angular_bounds_n = np.linspace(0.0, np.pi, num_in_half + 1)
        # angular boundaries on the positive side of the y axis, ordered
        # counterclockwise, including boundaries at 0 and pi
    angular_bounds_s = -angular_bounds_n[-2::-1]
        # angular boundaries on the negative side of the y axis, ordered
        # counterclockwise, *not* including the boundary at pi or 2pi
    angular_bounds = np.concatenate((angular_bounds_n, angular_bounds_s))
    angular_bounds = (angular_bounds + major_axis) % (np.pi*2)
    intervals = np.asarray(zip(angular_bounds[:-1], angular_bounds[1:]))
    intervals = intervals[:, np.newaxis, :]
        # add size-1 middle axis to indicate each bin is a single box
    return intervals


def partition_quadparity_folded(rad_interval, major_axis=None,
                                aspect_ratio=None):
    """
    Partition an annulus into angular bins that have parity across
    both axes.

    major_axis - float
        The major axis, as an angle counterclockwise from the
        x-axis, in radians. Bins will have parity across the major
        axis and the minor (major_axis + pi/2).
    aspect_ratio - float, default=1.5
        The target aspect ratio of the constructed bins, defined as
        angular_size/radial_size. The bins will have an aspect ratio
        no larger than the passed value, but it may be smaller.
    """
    unfolded_bins = partition_quadparity(rad_interval, major_axis=major_axis,
                                         aspect_ratio=aspect_ratio)
    num_unfolded_bins = unfolded_bins.shape[0]  # always even
    midpoint = num_unfolded_bins/2
    bins_n_of_ma = unfolded_bins[:midpoint]
    bins_s_of_ma = unfolded_bins[midpoint:][::-1, ...]
        # these are now ordered such that corresponding elements are
        # reflected pairs: bins_n_of_ma[j] is the reflection across
        # the major_axis of bins_s_of_ma[j]
    folded_bins = np.concatenate((bins_n_of_ma, bins_s_of_ma), axis=1)
    return folded_bins


def polar_threshold_binning(collection=None, coords=None, ids=None,
                            linear_scale=None, indexing_func=None,
                            combine_func=None, score_func=None,
                            threshold=None, step_size=None,
                            angle_partition_func=None):
    """
    ABSTRACT
    """
    coords = np.asarray(coords, dtype=float)
    ids = np.asarray(ids, dtype=int)
    linear_scale = float(linear_scale)
        # ADD SHAPE TESTING?
    radii = np.sqrt(np.sum(coords**2, axis=1))
    angles = np.arctan2(coords[:, 1], coords[:, 0])  # output in [-pi, pi]
    # find division between innermost bin boundary and solitary objects
    initial_scores = score_func(collection)
    passed = (initial_scores > threshold)
    central_obj_passed = passed[np.argmin(radii)]
    if not central_obj_passed:
        # there is no radius enclosing only above-threshold objects
        nobin_radius = 0.0
    elif np.all(passed):
        nobin_radius = np.max(radius) + 2*linear_scale
            # ensure that all object footprints are contained by
            # the nobin_radius, robust to roundoff errors
        grouped_ids = [[n] for n in ids]
        radial_bounds, angular_bounds = [], []
        return grouped_ids, [], []
    else:
        supremum = radii[~passed].min()
            # maximum radius below which no object require binning
        infimum = radii[radii < supremum].max()
            # minimum radius below which no object require binning - the
            # next-inward object to that identified in supremum
        nobin_radius = 0.5*(supremum + infimum)
            # average hopefully makes footprint inclusion robust to roundoff
    is_solitary = radii < nobin_radius
    solitary_ids = ids[is_solitary]
    # compute multi-object bins
    if step_size is None:
        # partition between every single spectrum to be binned
        radii_tobin = radii[~is_solitary]
        sorted_radii = np.sort(radii_tobin)
        midpoints = (sorted_radii[:-1] + sorted_radii[1:])*0.5
        final_radius = radii_tobin.max() + 2*linear_scale
        radial_partition = ([nobin_radius], midpoints, [final_radius])
        radial_partition = np.concatenate(radial_partition)
        step_size = np.median(midpoints[1:] - midpoints[:-1]) # typical gap
    else:
        # partition by passed step size
        step_size = float(step_size)
        upper_bound = radii.max() + 2*linear_scale + step_size
        radial_partition = np.arange(nobin_radius, upper_bound, step_size)
        final_radius = radial_partition.max() # ~ upper_bound - step_size
    # start binning loop
    starting_index = 0
    final_index = radial_partition.shape[0] - 1
    radial_bounds, angular_bounds, binned_ids = [], [], []
    while starting_index < final_index:
        lower_rad = radial_partition[starting_index]
        possible_upper_rads = radial_partition[(starting_index + 1):]
        for upper_iter, trial_upper_rad in enumerate(possible_upper_rads):
            rad_interval = [lower_rad, trial_upper_rad]
            try:
                angle_parition = angle_partition_func(rad_interval)
            except ValueError:
                break # invalid annulus - increase outer radius
            in_annulus = utl.in_linear_interval(radii, rad_interval)
            grouped_annular_ids = []
            for ang_intervals in angle_parition:
                in_arc_func = functools.partial(utl.in_periodic_interval,
                                                period=2*np.pi)
                in_arcs = utl.in_union_of_intervals(angles, ang_intervals,
                                                    inclusion=in_arc_func)
                in_bin = in_arcs & in_annulus
                ids_in_bin = ids[in_bin]
                objs_in_bin = indexing_func(collection, ids_in_bin)
                try:
                    combined_object = combine_func(objs_in_bin)
                    bin_score = score_func(combined_object)[0]
                        # combined_object is a collection with 1 element
                except ValueError, msg:
                    break # invalid bin - increase outer radius
                if bin_score < threshold:
                    break # bin too noise - increase outer radius
                grouped_annular_ids.append(ids_in_bin)  # bin accepted
            else:
                # this clause runs only if the above 'for' does not 'break'
                # i.e., all bins valid and pass
                binned_ids += grouped_annular_ids
                radial_bounds.append(rad_interval)
                angular_bounds.append(angle_parition)
                starting_index += upper_iter + 1 # ~ num radial steps taken
                break  # start new annulus
        else:
            # final annulus does not pass threshold - process outer objects
            # for now, discard outer objects
            break
    grouped_ids = [[id] for id in solitary_ids] + binned_ids
    return grouped_ids, radial_bounds, angular_bounds
