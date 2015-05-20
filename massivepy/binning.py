"""
This module contains functions for the spacial binning of IFU spectra,
and for the manipulation of sets of bins.
"""


import sys  # DEBUGGING

import numpy as np
np.seterr(all='raise')   # force numpy warnings to raise exceptions
np.seterr(under='warn')  # for all but underflow warnings (numpy
                         # version 1.8.2 raises spurious underflows
                         # on some masked array computations)

import utilities as utl


# def unfolded_partitioner(radii, angles, major_axis, aspect_ratio):
#     """
#     Partition fibers into angular bins symmetric across both axes.

#     aspect_ratio - float, default=1.5
#         The target aspect ratio of the constructed bins, defined as
#         angular_size/radial_size. The final bins will have an
#         aspect_ratio no larger than the passed value, but it may be
#         smaller as the bins is restricted to be in only one quadrant.
#     """
#     angles -= major_axis
#     outer_radius, inner_radius = radii.max(), radii.min()
#     delta_r = outer_radius - inner_radius
#     mid_r = 0.5*(outer_radius + inner_radius)
#     target_bin_arclength = delta_r*aspect_ratio
#     available_arclength = 0.5*np.pi*mid_r  # one quadrant
#     num_firstquad = int(available_arclength/target_bin_arclength)
#     if num_firstquad == 0:
#         raise ValueError
#     num_bins_north = 2*num_firstquad
#     angular_bounds_n = np.linspace(0.0, np.pi, num_bins_north + 1)
#         # angular boundaries on the positive side of the y axis,
#         # counterclockwise, including boundaries at 0 and pi
#     angular_bounds_s = -angular_bounds_n[-2::-1]
#         # angular boundaries on the negative side of the y axis,
#         # counterclockwise, *not* including the boundary at pi or 2pi
#     # angular_bounds_s[-1] = 2*np.pi
#     #     # shift will cause angular_bounds_s in [0, 2pi), i.e. the final bound
#     #     # is 0 - this will cause orderer issues, need it to be 2pi instead
#     angular_bounds = np.concatenate((angular_bounds_n, angular_bounds_s))
#     # sort fibers
#     # partition = []
#     # angle_iteration = zip(angular_bounds[:-1], angular_bounds[1:])
#     # for start, stop in angle_iteration:
#     #     angle_selector = ((start <= angles) & (angles < stop))
#     #     partition.append(fibers[angle_selector])
#     angular_bounds = (angular_bounds + major_axis) % (np.pi*2)
#     return angular_bounds

# above here untested

def partition_quadparity(rad_interval, major_axis=None, aspect_ratio=None):
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
    return angular_bounds



# def binfiber_polar(ifu_spectra, s2n_limit, angular_paritioner,
#                    step_size=None, compute_s2n=None,
#                    combine_spectra=inverse_variance):
#     """
#     Construct polar, spacial bins from the passed spectra, which each
#     bin meeting the passed S/N threshold. Individual spectra with S/N
#     above the threshold will be kept unbinned.

#     The bins are 'polar' in the sense that each bin is defined by a
#     minimum and maximum radius and angle. The plane is partitioned as
#     a series of concentric annuli, each subdivided by angle into the
#     final binning. The pattern of angular subdivision is arbitrary,
#     determined by a user-supplied function.

#     The binning algorithm is as follows:
#       1 - Find radius below which all fibers pass S/N threshold.
#           This is r_start. The fibers below r_start are left unbinned.
#       2 - Set inner and outer bin radii to r_in = r_out = r_start.
#       3 - Increment r_out one step outward.
#       4 - Partition the annulus r_in < r < r_out into angular bins.
#       5 - If any angular bin falls below the S/N threshold and r_out
#           remains less than the outermost spectrum, go to 3.
#         - If any angular bin falls below the S/N threshold but r_out
#           exceeds the outermost spectrum, end the binning. Discard the
#           unbinned outer spectra, i.e. those with radii currently
#           greater than r_in
#         - If all angular bins exceed S/N threshold, accept these bins
#           and continue with 6.
#       6 - r_out has by now been incremented to some r_out > r_in.
#           Generate more bins: increase r_in to r_in = r_out, go to 3.

#     Args:
#     ifu_spectra - IFUspectrum object
#         The spectra with corresponding spacial positions to bin.
#     s2n_limit - float
#         The S/N threshold above which to consider a bin valid.
#     angular_partitioner - function
#         The function that divides an annulus of spectra into angular
#         bins. It should have two arguments, the angles and radii of
#         the spectra to be partitioned, and should output the angular
#         boundaries. It may throw a ValueError to indicate that the
#         passed set of spectra cannot be acceptably partitioned.
#     step_size - float, defalt=None
#         The radial step size to use when incrementing the outer bin
#         radius. If not given, each step will be taken such that the
#         incremented bin includes the minimum possible nonzero number
#         of addition spectra, typically one.
#     compute_s2n - function
#         Function that accepts a spectrum, noise, and mask and returns
#         a single float S/N metric for the passes spectrum.
#     combine_spectra - function, default=inverse_variance
#         The function used to combine fiber spectra. This must accept
#         three inputs: spectrum, noise, and bad_data, each of which is
#         a 1d array. The output should be a 3 element list, the entries
#         of which are the combined spectrum, noise, and bad_data.

#     Returns:
#     """
#     num_spectra = ifu_spectra.spectrumset.num_spectra
#     spectra_ids = np.arange(num_spectra, dtype=int)  # integer identifiers
#     coords = ifu_spectra.coords  # Cartesian, shape (N, 2)
#     radii = np.sqrt(np.sum(coords**2, axis=1))
#     angles = np.arctan2(coords[:, 1], coords[:, 0])  # output in [-pi, pi]
#     outer_coords = coords[np.argmax(radii), :]
#     outer_footprint = ifu_spectra.footprint(outer_coords)
#     outer_circrad = utl.bounding_radius(outer_footprint, outer_coords)
#         # ~cirumradius of the footprint of the outermost spectrum
#     spectra = ifu_spectra.spectrumset.spectra
#     noises = ifu_spectra.spectrumset.metaspectra['noise']
#     bad_data = ifu_spectra.spectrumset.metaspectra['bad_data']
#     # find division between innermost bin boundary and solitary spectra
#     initial_s2n = compute_s2n(spectra, noises, bad_data)
#     s2n_passes = (initial_s2n > s2n_limit)
#     central_spectrum_passed = s2n_passes[np.argmin(radii)]
#     if not central_spectrum_passed:
#         # there is no radius enclosing only above-threshold spectra
#         nobin_radius = 0.0
#     elif np.all(s2n_passes):
#         nobin_radius = np.max(radius) + 2*outermost_circrad
#             # ensure that all spectra footprints are contained by
#             # the nobin_radius, robust to roundoff errors
#             # ADD RETURN STATE
#     else:
#         supremum = radii[~s2n_passes].min()
#             # maximum radius below which no spectra require binning
#         infimum = radii[radii < supremum].max()
#             # minimum radius below which no spectra require binning - the
#             # next-inward spectrum to that identified in supremum
#         nobin_radius = 0.5*(supremum + infimum)
#             # average hopefully makes footprint inclusion robust to roundoff
#     is_solitary = radii < nobin_radius
#     tobin_spectra_ids = spectra_ids[~is_solitary]
#     tobin_radii = radii[~is_solitary]
#     tobin_angles = angles[~is_solitary]
#     tobin_spectra = spectra[~is_solitary]
#     tobin_noises = noises[~is_solitary]
#     tobin_bad_data = bad_data[~is_solitary]
#     # compute multi-spectra bins
#     if step_size is None:
#         # partition between every single spectrum to be binned
#         sorted_tobin_radii = np.sort(tobin_radii)
#         midpoints = (sorted_tobin_radii[:-1] + sorted_tobin_radii[1:])*0.5
#         final_radius = tobin_radii.max() + 2*outer_circrad
#         radial_partition = ([nobin_radius], midpoints, [final_radius])
#         radial_partition = np.concatenate(radial_partition)
#         step_size = np.median(midpoints[1:] - midpoints[:-1]) # typical gap
#     else:
#         # partition by passed step size
#         step_size = float(step_size)
#         arange_lim = tobin_radii.max() + 2*outer_circrad + step_size
#         radial_partition = np.arange(nobin_radius, arange_lim, step_size)
#         final_radius = radial_partition.max() # ~ arange_lim - step_size
#     # start binning loop
#     starting_index = 0
#     final_index = radial_partition.shape[0] - 1
#     radial_bounds = []
#     angular_bounds = []
#     binned_spectra_ids = []
#     print "solitary radius", nobin_radius
#     print "outermost radius", radial_partition.max()
#     while starting_index < final_index:
#         print ""
#         print "new annulus"
#         lower_rad = radial_partition[starting_index]
#         possible_upper_rad = radial_partition[(starting_index + 1):]
#         for upper_iter, trial_upper_rad in enumerate(possible_upper_rad):
#             print "lower rad", lower_rad
#             print upper_iter, "trial upper rad:", trial_upper_rad
#             trial_radial_interval = [lower_rad, trial_upper_rad]
#             in_annulus = utl.in_linear_interval(tobin_radii,
#                                                 trial_radial_interval)
#             num_spectra_in_annulus = np.sum(in_annulus)
#             print "num spec in annulus", num_spectra_in_annulus
#             if num_spectra_in_annulus == 0:  # empty - increase outer radius
#                 continue
#             # attempt angular partition
#             dividers = angular_paritioner(tobin_radii[in_annulus],
#                                           tobin_angles[in_annulus])
#             # try:
#             #     dividers = angular_paritioner(tobin_radii[in_annulus],
#             #                                   tobin_angles[in_annulus])
#             # except ValueError:
#             #     # unable to partition annulus - increase radius
#             #     continue
#             # check S/N of angular bins
#             starting_angles = dividers.copy()
#             # ending_angles = np.concatenate((dividers[1:], dividers[-1:]))
#             # angle_intervals = zip(starting_angles, ending_angles)
#             angle_intervals = np.asarray(zip(dividers[:-1], dividers[1:]))
#             accepted_spectra_ids = [] # spectra ids in each valid bin
#             for angle_interval in angle_intervals:
#                 in_arc = utl.in_periodic_interval(tobin_angles,
#                                                   angle_interval,
#                                                   period=2*np.pi)
#                 in_trialbin = in_arc & in_annulus
#                 num_in_trialbin = in_trialbin.sum()
#                 if num_in_trialbin == 0: # empty - increase outer radius
#                     break
#                 trial_indivual_data = [tobin_spectra[in_trialbin, :],
#                                        tobin_noises[in_trialbin, :],
#                                        tobin_bad_data[in_trialbin, :]]
#                 trial_binned = combine_spectra(*trial_indivual_data)
#                     # outputs binned spectra, noise, bad_data - use for S/N
#                     # computation and discard, re-computing later, which is
#                     # cleaner and saves memory in exchange for small slowdown
#                 trial_s2n = compute_s2n(*trial_binned)
#                 if trial_s2n < s2n_limit: # too noisy - increase outer radius
#                     print "too noisy"
#                     break
#                 # angular section valid, save results
#                 accepted_spectra_ids.append(tobin_spectra_ids[in_trialbin])
#             else:
#                 print "accpted", angle_intervals.shape[0], "divisions"
#                 # this clause runs only if the above 'for' does not 'break'
#                 # i.e., angular binning was successful, all sections valid
#                 binned_spectra_ids.append(accepted_spectra_ids)
#                 radial_bounds.append(trial_radial_interval)
#                 angular_bounds.append(angle_intervals)
#                 starting_index += upper_iter + 1 # ~ num radial steps taken
#                 break  # start new annulus
#         else:
#             # final annulus does not pass threshold - process outer spectra
#             # for now, discard outer spectra
#             break
#     # DEVELOPMENT - start here, need to re-combine
#     # binning finished - package results
#     # solitary_spectra_ids = spectra_ids[is_solitary]
#     # solitary_spectra_ids = solitary_spectra_ids[:, np.newaxis]
#     #     # now has form of a list of lists of fiber ids
#     # all_binned_spectra_ids = np.concatenate((solitary_spectra_ids,
#     #                                          binned_spectra_ids))
#     # all_bined_spectra = np.concatenate((spectra[solitary_spectra_ids],
#     #                                     accepted_spectra))
#     # all_bined_noise = np.concatenate((noise[solitary_spectra_ids],
#     #                                   accepted_noise))
#     # all_bined_baddata = np.concatenate((baddata[solitary_spectra_ids],
#     #                                     accepted_baddata))

#     return spectra_ids[is_solitary], binned_spectra_ids, radial_bounds, angular_bounds



def polar_binning(collection=None, coords=None, ids=None,
                  linear_scale=None, indexing_func=None,
                  combine_func=None, score_func=None, threshold=None,
                  angle_partition_func=None, step_size=None):
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
        print 'new annulus'
        lower_rad = radial_partition[starting_index]
        possible_upper_rads = radial_partition[(starting_index + 1):]
        for upper_iter, trial_upper_rad in enumerate(possible_upper_rads):
            rad_interval = [lower_rad, trial_upper_rad]
            print rad_interval
            try:
                angle_parition = angle_partition_func(rad_interval)
            except ValueError:
                break # invalid annulus - increase outer radius
            angle_intervals = np.asarray(zip(angle_parition[:-1],
                                             angle_parition[1:]))
                # assuming angle_parition[0] = angle_parition[-1]
            in_annulus = utl.in_linear_interval(radii, rad_interval)
            grouped_annular_ids = []
            for ang_interval in angle_intervals:
                in_wedge = utl.in_periodic_interval(angles, ang_interval,
                                                    period=2*np.pi)
                in_bin = in_wedge & in_annulus
                print "in trial bin:", in_bin.sum()
                ids_in_bin = ids[in_bin]
                objs_in_bin = indexing_func(collection, ids_in_bin)
                try:
                    combined_object = combine_func(objs_in_bin)
                    bin_score = score_func(combined_object)[0]
                        # combined_object is a collection with 1 element
                except ValueError:
                    break # invalid bin - increase outer radius
                if bin_score < threshold:
                    print 'too noisy'
                    break # bin too noise - increase outer radius
                grouped_annular_ids.append(ids_in_bin)  # bin accepted
            else:
                print 'accepted'
                # this clause runs only if the above 'for' does not 'break'
                # i.e., all bins valid and pass
                binned_ids += grouped_annular_ids
                radial_bounds.append(rad_interval)
                angular_bounds.append(angle_intervals)
                starting_index += upper_iter + 1 # ~ num radial steps taken
                break  # start new annulus
        else:
            # final annulus does not pass threshold - process outer objects
            # for now, discard outer objects
            break
    grouped_ids = [[id] for id in solitary_ids] + binned_ids
    return grouped_ids, radial_bounds, angular_bounds
