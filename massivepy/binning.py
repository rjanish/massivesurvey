""" This module contains functions for the spacial binning of data. """


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
    Partition an annulus into angular sections that have parity across
    both axes, with each section and its mirror across the major-axis
    together considered one bin. This is equivalent to folding the
    bins made by partition_quadparity_folded across the major axis.

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
    Bin spacial data radially, according to a threshold score.

    This takes a set of data objects with the following properties:
    each object has an associated 2D coordinate, any number of objects
    may be combining into another object of the same type, and any
    single object may be assigned a numerical score. Objects will be
    grouped into polar bins (defined by an interval in radius and
    angle) such that the score of the combined object encompassing
    each bin exceeds the given threshold.

    The binning algorithm starts at the center, accumulating objects
    into bins until the threshold is met, and then starting a new bin.
    The procedure is schematically:
      0 all objects at the center that pass the threshold on their
        own are kept unbinned (this anticipates that the score will
        be increasing outward)
      1 A trial annulus is chosen with fixed inner radius.
      2 The objects in the trial annulus are are sub-divided by angle
        according to the given angle_partition_func.
      3 If all bins in the trial annulus pass, start over at 1 with a
        new annulus directly outside the previous one. If not, increase
        the radius of the annulus and restart at 2.
      4 The outermost set of bins will likely not pass the threshold,
        and are not included in the binning.

    Args:
    collection - non-specific type, required properties given below
        This is an container that holds all of the data objects to
        be binned. The size is assumed to be N below.
    coords - Nx2 arraylike
        These are 2D Cartesian coordinates associated with the data
        objects: coords[n, :] are the coordinates of collection[n]
    ids - length N iterable of integers
        These are integer id numbers of each object in collection.
    linear_scale - float
        This is a linear dimension associated with the spacial size
        of each object, used to ensure the final bin boundaries do
        not cut through objects that have some spacial extent.
    indexing_func - func
        This is a function which selects a subset of the data from
        collection based on id numbers. I.e., the call
        indexing_func(collection, [j, k, l]) will return a container
        of the same type as collection which holds the three objects
        with id's j, k, and l.
    combine_func - func
        A function which takes a collection as above and returns a new
        collection containing only one data item that corresponds to
        the combination of all data in the original collection.
    score_func - func
        This function takes a collection as above, and returns a list
        of scores for each object in the collection.
    threshold - float
        The score threshold above which a bin is considered valid.
    step_size - float
        The radial step size to take when enlarging bins. By default,
        each step will be made so as to include in the new bin the
        fewest possible number of new data objects.
    angle_partition_func - func
        A function that accepts as a single argument a radial interval
        [rmin, rmax] and returns a partition of that annulus into
        angular bins, each bin consisting of the union of a number of
        angular intervals.
        The output format is, for output = angle_partition_func(...):
          - output[0] is the first bin, output[1] the second, etc.
          - Each bin entry in output is itself a list of angular
            intervals: output[j] = [[a0, a1], [a2, a3], ...]
          - The bin associated with output[j] is the union of all
            angular intervals in output[j]

    Returns: grouped_ids, radial_bounds, angular_bounds
    grouped_ids - iterable
        This has one entry for each bin, with each entry listing the
        id numbers of the objects in that bin.
    radial_bounds - Mx2 arraylike
        The rmin, rmax boundaries of each annular section of bins.
        The size will be M = (total number of bins) - (number of center
        bins containing a single object)
    angular_bounds - length M iterable
        A list of angular partitions, of size M, with angular_bounds[j]
        specifying the partitioning of the annulus radial_bounds[j].
        Each entry angular_bounds[i] has the format returned by
        the angle_partition_func (see above).
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
                    break # bin too noisy - increase outer radius
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

def calc_bin_center(xs,ys,fluxes,bintype,ma=None,rmin=None):
    """
    Calculate the flux-weighted bin center for a single bin, given the
    coordinates of each fiber in the bin (xs,ys) and the flux for each
    fiber (fluxes). If the bin type is folded, reflect all points across
    ma (except single fiber bins within rmin) before binning. Return as 
    an array for convenience.
    Note that this computes the fluxweighted coordinates in cartesian
    coordinates, then simply converts to r and theta. Perhaps it would make
    sense to compute polar coordinates by directly finding the fluxweighted
    average radius and angle instead.
    """
    if bintype=='unfolded':
        pass
    elif bintype=='folded':
        ii = [] #List of fibers needing to be reflected
        if len(xs)==1 and np.sqrt(xs[0]**2+ys[0]**2) < rmin:
            pass
        else:
            ys_ma_line = xs*np.tan(ma)
            ii = np.where(ys < ys_ma_line)[0]
        #Math for reflecting point over line y = m*x:
        # xnew = A - x, ynew = A*m - y, where A = 2 (x + m*y) / (1 + m^2)
        A = 2*(xs[ii]+ys[ii]*np.tan(ma))/(1+np.tan(ma)**2)
        xs[ii] = A - xs[ii]
        ys[ii] = A*np.tan(ma) - ys[ii]
    else:
        raise Exception('Bin type must be folded or unfolded, try again.')
    total_flux = fluxes.sum()
    x_bin = np.sum(xs*fluxes)/total_flux
    y_bin = np.sum(ys*fluxes)/total_flux
    r_bin = np.sqrt(x_bin**2 + y_bin**2)
    th_bin = np.arctan2(y_bin, x_bin)
    return np.array([x_bin,y_bin,r_bin,th_bin])
