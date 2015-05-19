"""
This module contains functions for the spacial binning of IFU spectra,
and for the manipulation of sets of bins.
"""


import numpy as np
np.seterr(all='raise')   # force numpy warnings to raise exceptions
np.seterr(under='warn')  # for all but underflow warnings (numpy
                         # version 1.8.2 raises spurious underflows
                         # on some masked array computations)

import utilities as utl

# below here untested

def median_normalize(spectra, noise=None, mask=None, mask_fill=None):
    """
    Simultaneous normalization of multiple spectra with noises,
    including possible masking of bad pixels. Each spectrum/noise pair
    is normalized to have a spectrum median over wavelength of 1.

    Args:
    spectrum - 1d or 2d float array
        An array of spectra to normalize, with each row a spectrum
        and each spectrum identically sampled in wavelength. A 1d
        array can be used for only a single spectrum.
    noise - 1d or 2d float array, default=None
        The estimated noise in each spectral value. A 1d array can be
        used for only a single spectrum. Ignored if not used.
    mask - 1d or 2d boolean array, defualt=None
        Indicator of spectral values to ignore. Only spectrum values
        corresponding to a mask of False are included when computing
        the normalization median. Setting this to None is equivalent
        to passing a mask array of all False. A 1d array can be used
        for only a single spectrum.
    mask_fill - float, default=None
        A fill value to be used in the spectrum and/or noise all
        masked pixels. If None, the no fill is uses and the masked
        data are treated as if valid (though not included in median).

    Returns: normalized_spectra, normalized_noise (optional)
    normalized_spectra - 2d float array
        Normalized spectra in the same format as the passed spectra.
    normalized_noise - 2d float array
        Normalized noises in the same format as the passed noise. Only
        returned if called with a noise array.
    """
    spectra = np.array(spectra, dtype=float)
    if noise is not None:
        noise = np.array(noise, dtype=float)
    if mask is None:  # assume all values are good
        mask = np.zeros(spectra.shape, dtype=bool)
    else:
        mask = np.array(mask, dtype=bool)
    masked_spectra = np.ma.array(spectra, mask=mask)
    scale_factors = np.ma.median(masked_spectra, axis=-1).data
    all_masked = np.all(mask, axis=-1)
        # identifies all-bad fibers, use to set their fiducial scale to 1.0
        # to keep numbers well-behaved. Default would set to zero.
    if scale_factors.ndim > 0:
        scale_factors[all_masked, ...] = 1.0
    else:  # spectra is 1d and identifiers are scalar, cannot use indexing
        if all_masked:
            scale_factors = 1.0
    normed_spectra = (spectra.T/scale_factors).T
        # if spectra 2d, scale_factors 1d: divides each row by constant
        # if spectra 1d, scale_factors 0d: divides single vector by constant
    if mask_fill is not None:
        normed_spectra[mask] = mask_fill
    if noise is None:
        return normed_spectra
    else:  # normalize noise with same scale factor
        normed_noise = (noise.T/scale_factors).T
        if mask_fill is not None:
            normed_noise[mask] = mask_fill
        return normed_spectra, normed_noise


def inverse_variance(spectra, noise, mask=None, mask_fill=None):
    """
    Combine multiple spectra with noise and possible masking, using an
    inverse variance weighted mean at each wavelength.

    Args:
    spectrum - 2d float array
        An array of spectra to combine, with each row a spectrum
        and each spectrum identically sampled in wavelength. No
        normalization is assumed.
    noise - 2d float array
        The estimated standard deviation of the noise in each spectral
        value. Spectra will be weighted by noise^(-2).
    mask - 2d boolean array, defualt=None
        Indicator of spectral values to ignore. At each wavelength,
        only spectrum values corresponding to a mask of False are
        included in the weighted mean.
    mask_fill - float, default=None
        If mask_fill is set and for some wavelengths all input spectra
        are masked, then the corresponding value in the final spectrum
        will be filled with mask_fill. Otherwise, the output is given
        by the weighted mean of the inputs as though they were valid.

    Returns: mean_spectrum, mean_noise, mean_mask
    mean_spectrum - 1d float array
        Combined and normalized spectrum
    mean_noise - 1d float array
        Combined and normalized standard deviation of the noise
    mean_mask - 1d boolean array
        Indicator of remaining, : masked pixels, i.e. those wavelengths
        for which all input data was masked.
    """
    spectra = np.array(spectra, dtype=float)
    noise = np.array(noise, dtype=float)
    if mask is not None:
        mask = np.array(mask, dtype=bool)
    normed_spectra, normed_noise = median_normalize(spectra, noise, mask)
    relative_weights = 1.0/normed_noise**2  # inverse variance weighting
    relative_weights[mask] = 0.0  # no masked pixels in weight normalization
    total_weight = relative_weights.sum(axis=0)
    all_masked = np.all(mask, axis=0)
    total_weight[all_masked] = 1.0
        # set nonzero fiducial total weight of wavelengths with no un-masked
        # values to avoid division errors; normalized weight is still zero
    weights = relative_weights/total_weight  # divide each column by constant
    [mean_spectrum, mean_noise,
     mean_mask, clipped] = clipped_mean(normed_spectra, weights=weights,
                                        noise=normed_noise, mask=mask)
    mean_spectrum, mean_noise = median_normalize(mean_spectrum, mean_noise,
                                                  mean_mask, mask_fill)
        # median_normalize will apply masked filling if needed
    return mean_spectrum, mean_noise, mean_mask


def unfolded_partitioner(radii, angles, major_axis, aspect_ratio):
    """
    Partition fibers into angular bins symmetric across both axes.

    aspect_ratio - float, default=1.5
        The target aspect ratio of the constructed bins, defined as
        angular_size/radial_size. The final bins will have an
        aspect_ratio no larger than the passed value, but it may be
        smaller as the bins is restricted to be in only one quadrant.
    """
    angles -= major_axis
    outer_radius, inner_radius = radii.max(), radii.min()
    delta_r = outer_radius - inner_radius
    mid_r = 0.5*(outer_radius + inner_radius)
    target_bin_arclength = delta_r*aspect_ratio
    available_arclength = 0.5*np.pi*mid_r  # one quadrant
    num_firstquad = int(available_arclength/target_bin_arclength)
    if num_firstquad == 0:
        raise ValueError
    num_bins_north = 2*num_firstquad
    angular_bounds_n = np.linspace(0.0, np.pi, num_bins_north + 1)
        # angular boundaries on the positive side of the y axis,
        # counterclockwise, including boundaries at 0 and pi
    angular_bounds_s = -angular_bounds_n[-2::-1]
        # angular boundaries on the negative side of the y axis,
        # counterclockwise, *not* including the boundary at pi or 2pi
    # angular_bounds_s[-1] = 2*np.pi
    #     # shift will cause angular_bounds_s in [0, 2pi), i.e. the final bound
    #     # is 0 - this will cause orderer issues, need it to be 2pi instead
    angular_bounds = np.concatenate((angular_bounds_n, angular_bounds_s))
    # sort fibers
    # partition = []
    # angle_iteration = zip(angular_bounds[:-1], angular_bounds[1:])
    # for start, stop in angle_iteration:
    #     angle_selector = ((start <= angles) & (angles < stop))
    #     partition.append(fibers[angle_selector])
    angular_bounds = (angular_bounds + major_axis) % (np.pi*2)
    return angular_bounds

# above here untested


def mean_s2n(data, noise, bad_data):
    """
    For a set of 1D measurements, compute the mean S/N of each
    measurement. The mean is taken uniform over the dependent
    variable, which is assumed to be the final dimension of the input.
    """
    masked_data = np.ma.array(data, mask=bad_data)
    masked_noise = np.ma.array(noise, mask=bad_data)
    return np.ma.mean(masked_data/masked_noise, axis=-1).data


def binfiber_polar(ifu_spectra, s2n_limit, angular_paritioner,
                   step_size=None, compute_s2n=None,
                   combine_spectra=inverse_variance):
    """
    Construct polar, spacial bins from the passed spectra, which each
    bin meeting the passed S/N threshold. Individual spectra with S/N
    above the threshold will be kept unbinned.

    The bins are 'polar' in the sense that each bin is defined by a
    minimum and maximum radius and angle. The plane is partitioned as
    a series of concentric annuli, each subdivided by angle into the
    final binning. The pattern of angular subdivision is arbitrary,
    determined by a user-supplied function.

    The binning algorithm is as follows:
      1 - Find radius below which all fibers pass S/N threshold.
          This is r_start. The fibers below r_start are left unbinned.
      2 - Set inner and outer bin radii to r_in = r_out = r_start.
      3 - Increment r_out one step outward.
      4 - Partition the annulus r_in < r < r_out into angular bins.
      5 - If any angular bin falls below the S/N threshold and r_out
          remains less than the outermost spectrum, go to 3.
        - If any angular bin falls below the S/N threshold but r_out
          exceeds the outermost spectrum, end the binning. Discard the
          unbinned outer spectra, i.e. those with radii currently
          greater than r_in
        - If all angular bins exceed S/N threshold, accept these bins
          and continue with 6.
      6 - r_out has by now been incremented to some r_out > r_in.
          Generate more bins: increase r_in to r_in = r_out, go to 3.

    Args:
    ifu_spectra - IFUspectrum object
        The spectra with corresponding spacial positions to bin.
    s2n_limit - float
        The S/N threshold above which to consider a bin valid.
    angular_partitioner - function
        The function that divides an annulus of spectra into angular
        bins. It should have two arguments, the angles and radii of
        the spectra to be partitioned, and should output the angular
        boundaries. It may throw a ValueError to indicate that the
        passed set of spectra cannot be acceptably partitioned.
    step_size - float, defalt=None
        The radial step size to use when incrementing the outer bin
        radius. If not given, each step will be taken such that the
        incremented bin includes the minimum possible nonzero number
        of addition spectra, typically one.
    compute_s2n - function
        Function that accepts a spectrum, noise, and mask and returns
        a single float S/N metric for the passes spectrum.
    combine_spectra - function, default=inverse_variance
        The function used to combine fiber spectra. This must accept
        three inputs: spectrum, noise, and bad_data, each of which is
        a 1d array. The output should be a 3 element list, the entries
        of which are the combined spectrum, noise, and bad_data.

    Returns:
    """
    num_spectra = ifu_spectra.spectrumset.num_spectra
    spectra_ids = np.arange(num_spectra, dtype=int)  # integer identifiers
    coords = ifu_spectra.coords  # Cartesian, shape (N, 2)
    radii = np.sqrt(np.sum(coords**2, axis=1))
    angles = np.arctan2(coords[:, 1], coords[:, 0])  # output in [-pi, pi]
    outer_coords = coords[np.argmax(radii), :]
    outer_footprint = ifu_spectra.footprint(outer_coords)
    outer_circrad = utl.bounding_radius(outer_footprint, outer_coords)
        # ~cirumradius of the footprint of the outermost spectrum
    spectra = ifu_spectra.spectrumset.spectra
    noises = ifu_spectra.spectrumset.metaspectra['noise']
    bad_data = ifu_spectra.spectrumset.metaspectra['bad_data']
    # find division between innermost bin boundary and solitary spectra
    initial_s2n = compute_s2n(spectra, noises, bad_data)
    s2n_passes = (initial_s2n > s2n_limit)
    central_spectrum_passed = s2n_passes[np.argmin(radii)]
    if not central_spectrum_passed:
        # there is no radius enclosing only above-threshold spectra
        nobin_radius = 0.0
    elif np.all(s2n_passes):
        nobin_radius = np.max(radius) + 2*outermost_circrad
            # ensure that all spectra footprints are contained by
            # the nobin_radius, robust to roundoff errors
            # ADD RETURN STATE
    else:
        supremum = radii[~s2n_passes].min()
            # maximum radius below which no spectra require binning
        infimum = radii[radii < supremum].max()
            # minimum radius below which no spectra require binning - the
            # next-inward spectrum to that identified in supremum
        nobin_radius = 0.5*(supremum + infimum)
            # average hopefully makes footprint inclusion robust to roundoff
    is_solitary = radii < nobin_radius
    tobin_spectra_ids = spectra_ids[~is_solitary]
    tobin_radii = radii[~is_solitary]
    tobin_angles = angles[~is_solitary]
    tobin_spectra = spectra[~is_solitary]
    tobin_noises = noises[~is_solitary]
    tobin_bad_data = bad_data[~is_solitary]
    # compute multi-spectra bins
    if step_size is None:
        # partition between every single spectrum to be binned
        sorted_tobin_radii = np.sort(tobin_radii)
        midpoints = (sorted_tobin_radii[:-1] + sorted_tobin_radii[1:])*0.5
        final_radius = tobin_radii.max() + 2*outer_circrad
        radial_partition = ([nobin_radius], midpoints, [final_radius])
        radial_partition = np.concatenate(radial_partition)
        step_size = np.median(midpoints[1:] - midpoints[:-1]) # typical gap
    else:
        # partition by passed step size
        step_size = float(step_size)
        arange_lim = tobin_radii.max() + 2*outer_circrad + step_size
        radial_partition = np.arange(nobin_radius, arange_lim, step_size)
        final_radius = radial_partition.max() # ~ arange_lim - step_size
    # start binning loop
    starting_index = 0
    final_index = radial_partition.shape[0] - 1
    radial_bounds = []
    angular_bounds = []
    binned_spectra_ids = []
    print "solitary radius", nobin_radius
    print "outermost radius", radial_partition.max()
    while starting_index < final_index:
        print ""
        print "new annulus"
        lower_rad = radial_partition[starting_index]
        possible_upper_rad = radial_partition[(starting_index + 1):]
        for upper_iter, trial_upper_rad in enumerate(possible_upper_rad):
            print "lower rad", lower_rad
            print upper_iter, "trial upper rad:", trial_upper_rad
            trial_radial_interval = [lower_rad, trial_upper_rad]
            in_annulus = utl.in_linear_interval(tobin_radii,
                                                trial_radial_interval)
            num_spectra_in_annulus = np.sum(in_annulus)
            print "num spec in annulus", num_spectra_in_annulus
            if num_spectra_in_annulus == 0:  # empty - increase outer radius
                continue
            # attempt angular partition
            dividers = angular_paritioner(tobin_radii[in_annulus],
                                          tobin_angles[in_annulus])
            # try:
            #     dividers = angular_paritioner(tobin_radii[in_annulus],
            #                                   tobin_angles[in_annulus])
            # except ValueError:
            #     # unable to partition annulus - increase radius
            #     continue
            # check S/N of angular bins
            starting_angles = dividers.copy()
            # ending_angles = np.concatenate((dividers[1:], dividers[-1:]))
            # angle_intervals = zip(starting_angles, ending_angles)
            angle_intervals = np.asarray(zip(dividers[:-1], dividers[1:]))
            accepted_spectra_ids = [] # spectra ids in each valid bin
            for angle_interval in angle_intervals:
                in_arc = utl.in_periodic_interval(tobin_angles,
                                                  angle_interval,
                                                  period=2*np.pi)
                in_trialbin = in_arc & in_annulus
                num_in_trialbin = in_trialbin.sum()
                if num_in_trialbin == 0: # empty - increase outer radius
                    break
                trial_indivual_data = [tobin_spectra[in_trialbin, :],
                                       tobin_noises[in_trialbin, :],
                                       tobin_bad_data[in_trialbin, :]]
                trial_binned = combine_spectra(*trial_indivual_data)
                    # outputs binned spectra, noise, bad_data - use for S/N
                    # computation and discard, re-computing later, which is
                    # cleaner and saves memory in exchange for small slowdown
                trial_s2n = compute_s2n(*trial_binned)
                if trial_s2n < s2n_limit: # too noisy - increase outer radius
                    print "too noisy"
                    break
                # angular section valid, save results
                accepted_spectra_ids.append(tobin_spectra_ids[in_trialbin])
            else:
                print "accpted", angle_intervals.shape[0], "divisions"
                # this clause runs only if the above 'for' does not 'break'
                # i.e., angular binning was successful, all sections valid
                binned_spectra_ids.append(accepted_spectra_ids)
                radial_bounds.append(trial_radial_interval)
                angular_bounds.append(angle_intervals)
                starting_index += upper_iter + 1 # ~ num radial steps taken
                break  # start new annulus
        else:
            # final annulus does not pass threshold - process outer spectra
            # for now, discard outer spectra
            break
    # DEVELOPMENT - start here, need to re-combine
    # binning finished - package results
    # solitary_spectra_ids = spectra_ids[is_solitary]
    # solitary_spectra_ids = solitary_spectra_ids[:, np.newaxis]
    #     # now has form of a list of lists of fiber ids
    # all_binned_spectra_ids = np.concatenate((solitary_spectra_ids,
    #                                          binned_spectra_ids))
    # all_bined_spectra = np.concatenate((spectra[solitary_spectra_ids],
    #                                     accepted_spectra))
    # all_bined_noise = np.concatenate((noise[solitary_spectra_ids],
    #                                   accepted_noise))
    # all_bined_baddata = np.concatenate((baddata[solitary_spectra_ids],
    #                                     accepted_baddata))

    return spectra_ids[is_solitary], binned_spectra_ids, radial_bounds, angular_bounds

