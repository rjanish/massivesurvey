"""
This module performs spacial binning of ifu data and contains
functions for working with previously binned data.
"""

import numpy as np
np.seterr(all='raise')   # force numpy warnings to raise exceptions
np.seterr(under='warn')  # for all but underflow warnings (numpy
                         # version 1.8.2 raises spurious underflows
                         # on some masked array computations)


def principal_value_shift(data, lower=0.0, period=2*np.pi):
    """
    Shift input data by integer multiples of period to
    satisfy lower < data < lower + period
    """
    return (data % period) + lower


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


def clipped_mean(data, weights=None, noise=None, mask=None, fill_value=None,
                clip=5, max_iters=5, max_fractional_remove=0.02,
                converge_fraction=0.02):
    """
    Compute a clipped, weighted mean of each column in the passed 2d
    array.  This is the weighted mean excluding any data points that
    differ from the unweighted median by greater than the passed clip
    factor times the standard deviation. This is iterative.

    Args:
    data - 2d ndarray
        Array for which the clipped, weighted mean of
        each column will be computed
    weights - 1d or 2d ndarray
        The weighting factors to be used.  Will be normalized so that
        each column of weights sums to 1.  If a 1d array is passed,
        the same weights will be used for each column. If unspecified,
        then uniform weights are used.
    noise - 2d ndarray, defalt=None
        The estimated noise in the passed data, will be ignored if not
        passed or combined in quadrature if an array is given
    fill_value - float, default=None
        If given, will be used to fill all mask values in the final
        output, otherwise masked values are allowed to float.
    clip - float, default=3
        All data differing from the unweighted median by more than
        clip*standard_deviation will be ignored
    max_iters - int, default=10
        maximum number of iterations before clipping is halted
    max_fractional_remove - float, default=0.02
        maximum fraction number of data points
        that can be clipped before clipping is halted
    mask - 2d boolean ndarray
        True for any pixels to be ignored in the computation.

    Returns: clipped_data_mean, clipped_data_noise (optional),
             mean_mask, clipped
    clipped_data_mean - 1d ndarray
        The clipped, weighted mean of each column of data
    clipped_data_noise - 1d ndarray
        The noise estimate in the clipped mean of each column, only
        returned if a noise array is passed
    mean_mask - 1d ndarray
        The masking array for the mean data, indicated any columns
        for which all values were either masked or clipped
    clipped_points - 2d boolean ndarray
        An array of the same size as the input data, with a True for
        every data point that was clipped
    """
    data = np.array(data, dtype=float)
    if noise is not None:
        noise = np.array(noise, dtype=float)
        normed_data, normed_noise = median_normalize(data, noise=noise,
                                                     mask=mask)
    else:
        normed_data = median_normalize(data, mask=mask)
    if mask is None:
        masked = np.zeros(data.shape, dtype=bool)
    else:
        masked = np.array(mask, dtype=bool)
    total_num_points = float(data.size)  # float for fractional divisions
    normed_masked_data = np.ma.array(normed_data, mask=masked)
    clipped = np.zeros(data.shape, dtype=bool)
    # compute clipping
    for iter in xrange(max_iters):
        sigma = np.ma.std(normed_masked_data, axis=0)
        central = np.ma.median(normed_masked_data, axis=0)
        distance = np.ma.absolute(normed_masked_data - central)/sigma
            # default broadcasting is to copy vector along each row
        new_clipped = (distance > clip).data
            # a non-masked array, any data already masked in distance are set
            # False by default in size compare - this finds new clipped only
        num_old_nonclipped = np.sum(~clipped)
        clipped = clipped | new_clipped  # all clipped points
        normed_masked_data.mask = clipped | masked  # actual clipping
        total_frac_clipped = np.sum(clipped)/total_num_points
        delta_nonclipped = np.absolute(np.sum(~clipped) - num_old_nonclipped)
        delta_frac_nonclipped = delta_nonclipped/float(num_old_nonclipped)
        if ((delta_frac_nonclipped <= converge_fraction) or  # convergence
            (total_frac_clipped > max_fractional_remove)):
            break
    # compute mean
    bad_pixels = masked | clipped
    if weights is None:
        weights = np.ones(data.shape)
    else:
        weights = np.array(weights, dtype=float)
    if weights.ndim == 1:
        weights = inflate(weights, 'v', masked_data.shape[1])
    weights[bad_pixels] = 0.0  # do not include clipped or masked in norm

    total_weight = weights.sum(axis=0)
    all_bad = np.all(bad_pixels, axis=0)
    total_weight[all_bad] = 1.0
        # set nonzero fiducial total weight for wavelengths with no un-masked
        # values to avoid division errors; normalized weight is still zero
    weights = weights/total_weight  # divides each col by const
    clipped_data_mean = np.ma.sum(normed_masked_data*weights, axis=0).data
    mean_mask = np.all(bad_pixels, axis=0)
    if fill_value is not None:
        clipped_data_mean[mean_mask] == fill_value
    if noise is not None:
        normed_masked_noise = np.ma.masked_array(normed_noise,
                                                 mask=bad_pixels)
        clipped_variance = np.ma.sum((normed_masked_noise*weights)**2,
                                     axis=0).data
        clipped_data_noise = np.sqrt(clipped_variance)
        if fill_value is not None:
            clipped_data_noise[mean_mask] == fill_value
        return clipped_data_mean, clipped_data_noise, mean_mask, clipped
    else:
        return clipped_data_mean, mean_mask, clipped


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


def folded_partitioner(fibers, angles, radii, aspect_ratio):
    """
    Partition fibers into angular bins folded about the x axis.


    aspect_ratio - float, default=1.5
        The target aspect ratio of the constructed bins, defined as
        angular_size/radial_size. The final bins will have an
        aspect_ratio no larger than the passed value, but it may be
        smaller as the bins is restricted to be in only one quadrant.
    """
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
        # angular boundaries on the north side of the major axis,
        # including boundaries at 0 and pi
    # sort fibers
    partition = []
    angle_iteration = zip(angular_bounds_n[:-1], angular_bounds_n[1:])
    for start, stop in angle_iteration:
        angle_selector_plus = ((start <= angles) & (angles < stop))
        reflected_start = 2*np.pi - stop
        reflected_stop = 2*np.pi - start
        angle_selector_minus = ((reflected_start <= angles) &
                                (angles < reflected_stop))
        angle_selector = angle_selector_plus | angle_selector_minus
        partition.append(fibers[angle_selector])
    return partition, angular_bounds_n


def unfolded_partitioner(fibers, angles, radii, aspect_ratio):
    """
    Partition fibers into angular bins symmetric across both axes.

    aspect_ratio - float, default=1.5
        The target aspect ratio of the constructed bins, defined as
        angular_size/radial_size. The final bins will have an
        aspect_ratio no larger than the passed value, but it may be
        smaller as the bins is restricted to be in only one quadrant.
    """
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
        # counterclockwise, *not* including the boundary at pi
    angular_bounds_s = principal_value_shift(angular_bounds_s,
                                             lower=0.0, period=2*np.pi)
    angular_bounds_s[-1] = 2*np.pi
        # shift will cause angular_bounds_s in [0, 2pi), i.e. the final bound
        # is 0 - this will cause orderer issues, need it to be 2pi instead
    angular_bounds = np.concatenate((angular_bounds_n, angular_bounds_s))
    # sort fibers
    partition = []
    angle_iteration = zip(angular_bounds[:-1], angular_bounds[1:])
    for start, stop in angle_iteration:
        angle_selector = ((start <= angles) & (angles < stop))
        partition.append(fibers[angle_selector])
    return partition, angular_bounds


def construct_polar_multifiber_bins(fibers, spectra, noise, mask, radius,
                                    angle, step_size=None, starting_radius=0.0,
                                    angular_paritioner=folded_partitioner,
                                    s2n_limit=20, combination=inverse_variance,
                                    major_axis=0.0):
    """
    Divides a set of ifu fibers into a polar grid of bins with parity
    across the minor axis and folding across the major axis, ensuring
    that the folded spectrum of each bin passes a s/n threshold. Bins
    are created from the center outwards, with any excess fibers that
    cannot be binned below threshold discarded.

    Args:
    fibers - 1d array
        Fiber numbers identifying each fiber to be binned
    spectra - 2d array
        2d array with a row for each fiber, where each row is the
        spectrum associated with that fiber.  Order of rows is
        assumed to match the identifiers given in fibers, and all
        spectra as assumed to be identically sampled over wavelength
    noise - 2d array
        Array with same format as spectra, giving the estimated
        one standard deviation noise level in spectra.
    mask - 2d array
        Array with same format as spectra, identifying any bad
        pixels with True.  These pixels will be ignored.
    radius - 1d array
        Array with the same format as fibers, giving the radial
        position of each fiber.
    angle - 1d array
        Array with the same format as fibers, giving the angular
        position in radians of each fiber.
    major_axis - float, default=0
        The angle to the major axis measured from E to N in radians.
        This is not the conventional position angle, which is related
        by: major_axis = pi/2 - position_angle
    step_size - float, defalt=None
        The radial step size to take when increasing the radius of
        a bin due to low s/n. If not given, each step will be taken
        so as to include exactly one more fiber in the annular bin.
    starting_radius - float, default=0.0
        Radius at which to start the binning.
    s2n_limit float, default=20
        The s/n threshold above which to consider a bin valid. The
        default is 20, which is used for MASSIVE folded bins.
    combination - function, default=inverse_variance
        A function to be used to combine fiber spectra, must take
        the inputs spectra, noise, and mask, in that order, and then
        return a combined spectrum, noise, and mask, also in that
        order, as the elements of an iterable of length 3.

    Returns: binned_fibers, binned_data, annular_sets
    binned_fibers - list
        A list of lists, with a sublist for each bin giving the fiber
        numbers of the fibers included in that bin
    binned_data - list
        A list with an entry for each bin, ordered identical to
        binned_fibers. Each entry is a 2d array giving the bin's
        spectral data: the first row is the normalized spectrum,
        the second the noise estimate, and the third a masking array.
    annular_sets - list
        A list giving the polar boundaries of the bins. Each bin is
        two disjoint areas, and this gives the boundaries for the
        binned region on the northern side of the major axis. The
        second region is the reflection of the given region across
        the major axis. This list contains an entry for each annular
        set of bins, which in turn contains the radial and angular
        boundaries of that set of bins in the format:
        [[rin, rout], [(a0, a1), (a1, a2), (a2, a3), ... ]].
    
    Bin formating conventions:
    For the bin in the nth radial position and the mth angular
    position (n, m starting at 0) on the north side of the major axis,
    that bin is identified as bin number q = n + m and has the
    properties:
      - the fibers in bin q are binned_fibers[q]
      - the spectrum of bin q is binned_data[q][0, :]
      - the noise in the spectrum of bin q is binned_data[q][1, :]
      - the masked wavelength pixels in bin q are binned_data[q][2, :]
      - the radial bounds of the north section of bin q
        are annular_sets[n][0]
      - the angular bounds of the north section of bin q
        are annular_sets[n][1][m]
    """
    # set radial partition
    if step_size is None:
        sorted_radius = np.sort(radius)
        midpoints = (sorted_radius[:-1] + sorted_radius[1:])*0.5
        above_starting = midpoints[midpoints > starting_radius]
        typical_gap = np.median(midpoints[1:] - midpoints[:-1])
        final_radius = radius.max() + typical_gap
        radial_partition = ([starting_radius] + above_starting.tolist() +
                            [final_radius])
    else:
        step_size = float(step_size)
        max_boundary_radius = radius.max() + step_size
        radial_partition = np.arange(starting_radius, max_boundary_radius,
                                     step_size)
    # binning
    starting_index = 0
    final_index = len(radial_partition) - 1
    binned_fibers, binned_data, annular_sets = [], [], []
    while starting_index < final_index:
        lower_radius = radial_partition[starting_index]
        bin_edges = radial_partition[(starting_index + 1):]
            # possible upper bin boundaries
        for outer_iter, outer_radius in enumerate(bin_edges):
            # short for an empty annular section
            fibers_in_annulus = ((lower_radius <= radius) &
                                 (radius < outer_radius))
            num_fibers_in_annulus = np.sum(fibers_in_annulus)
            if (num_fibers_in_annulus == 0):
                continue  # increase outer radius of annulus
            # make angular partition (angles are in frame: major_axis = 0)
            try:
                fibers_in_trial, bounds = (
                    angular_paritioner(fibers[fibers_in_annulus],
                                       angle[fibers_in_annulus],
                                       radius[fibers_in_annulus]))
            except ValueError:
                # the annulus unable to be partitioned - increase radius
                continue
            trial_selectors = []
            for trial_fibers in fibers_in_trial:
                trial_selector = np.zeros(fibers.shape, dtype=bool)
                for fiber_index, fiber_num in enumerate(fibers):
                    if fiber_num in trial_fibers:
                        trial_selector[fiber_index] = True
                trial_selectors.append(trial_selector)
            trial_binned_data = []
            trial_binned_fibers = []
            for trial_selector in trial_selectors:
                num_trial_fibers = np.sum(trial_selector)
                # short for an empty angular section
                if num_trial_fibers == 0:
                    break  # increase outer radius of annulus
                trial_combined = combination(spectra[trial_selector, :],
                                             noise[trial_selector, :],
                                             mask[trial_selector, :])
                trial_spectra, trial_noise, trial_mask = trial_combined
                masked_trial_spectra = np.ma.array(trial_spectra,
                                                   mask=trial_mask)
                masked_trial_noise = np.ma.array(trial_noise,
                                                 mask=trial_mask)
                full_trial_s2n = masked_trial_spectra/masked_trial_noise
                trial_s2n = np.ma.mean(full_trial_s2n)
                # short for a too-noisy angular section
                if (trial_s2n < s2n_limit):
                    break  # increase outer radius of annulus
                # angular section valid
                trial_binned_fibers.append(fibers[trial_selector].tolist())
                trial_binned_data.append(np.array(trial_combined))
            else:
                # angular binning was successful, all sections valid
                binned_fibers += trial_binned_fibers
                binned_data += trial_binned_data
                radial_bounds = [lower_radius, outer_radius]
                sky_angular_bounds_n = list(bounds + major_axis)
                    # angular boundaries in the frame with E = 0 and N = 90
                sky_angular_pairs = zip(sky_angular_bounds_n[:-1],
                                        sky_angular_bounds_n[1:])
                annular_sets.append([radial_bounds, sky_angular_pairs])
                num_radial_steps = outer_iter + 1
                starting_index += num_radial_steps
                break  # start new annular bin
        else:
            # final annulus does not meet binning criteria
            # handle outer fibers.
            break
    return binned_fibers, binned_data, annular_sets


def find_single_fiber_region(spectra, noise, mask, radius,
                             fiber_radius=2.08, s2n_limit=20):
    """
    Compute the maximum radius below which all fibers are above a
    given signal-to-noise threshold.

    To avoid future rounding issues, the radius is forced to run
    between fibers. If no fibers are above the threshold, the radius
    is returned zero, and if all fibers are above the threshold, the
    returned radius is the smallest radius that enclosed the entire
    field of fibers with an outer cushion of one fiber radius.

    Args:
    
    spectra - 2d float array
        2d array with a row for each fiber, where each row is the
        spectrum associated with that fiber.  Order of rows is
        assumed to match that of the identifiers in fibers_id, and all
        spectra as assumed to be identically sampled over wavelength
    noise - 2d float array
        Array with same format as spectra, giving the estimated
        one standard deviation noise level in spectra.
    mask - 2d bool array
        Array with same format as spectra, identifying any bad
        pixels with True.  These pixels will be ignored.
    radius - 2D arraylike
        The radial position of each fiber
    fiber_radius - float, default=2.08
        The radius of each fiber. The default is 2.08, which
        is the arcsec radius of the VIRUS-P fibers.
    s2n_limit float, default=20
        The s/n threshold above which to consider a bin valid. The
        default of 20 is used for MASSIVE folded bins.

    Returns: single_fiber_radius
    single_fiber_radius - float
        Boundary below which all fibers pass the s/n threshold.
    """
    masked_spectra = np.ma.array(spectra, mask=mask)
    masked_noise = np.ma.array(noise, mask=mask)
    s2n = np.ma.mean(masked_spectra/masked_noise, axis=1).data
    s2n_passes = (s2n > s2n_limit)
    central_fiber_passed = s2n_passes[np.argmin(radius)]
    if not central_fiber_passed:
        # there can be no radius enclosing only above-threshold fibers
        single_fiber_boundary = 0.0
    elif np.all(s2n_passes):
        single_fiber_boundary = np.max(radius) + 2*fiber_radius
    else:
        supremum = radius[~s2n_passes].min()
            # maximum radius below which no fibers require binning:
            # located at center of the innermost below-threshold fiber
        infimum = radius[radius < supremum].max()
            # minimum radius below which no fibers require binning:
            # located at the center of the fiber that is the next-most-inner
            # fiber to the fiber identified in supremum
        single_fiber_boundary = 0.5*(supremum + infimum)
    return single_fiber_boundary


def construct_polar_bins(spectra, noise, mask, coords, major_axis=0,
                         fiber_radius=2.08, s2n_limit=20, step_size=None,
                         combination=inverse_variance,
                         angular_paritioner=folded_partitioner,
                         fibers_id=None):
    """
    Construct polar bins from the passed spectra. Individual spectra
    that exceed the s/n threshold will be kept un-binned. The multi-
    fiber bins will be symmetric across both the major and minor axes.
    The multi-fiber bins can be folded across the major axis.

    
    fibers_id - 1d int array, default=[0, 1, 2, ...]
        Fiber numbers identifying each fiber to be binned
    spectra - 2d float array
        2d array with a row for each fiber, where each row is the
        spectrum associated with that fiber.  Order of rows is
        assumed to match that of the identifiers in fibers_id, and all
        spectra as assumed to be identically sampled over wavelength
    noise - 2d float array
        Array with same format as spectra, giving the estimated
        one standard deviation noise level in spectra.
    mask - 2d bool array
        Array with same format as spectra, identifying any bad
        pixels with True.  These pixels will be ignored.
    coords - 2d float array
        2d array with each row corresponding to a fiber, listed
        in the same order as fibers_id. Each row contains the
        fiber's two Cartesian position coordinates.
    major_axis - float, default=0
        The angle to the major axis measured from E to N in radians.
        This is NOT the conventional position angle, which is related
        by: major_axis = pi/2 - position_angle
    step_size - float, defalt=None
        The radial step size to take when increasing the radius of
        a bin due to low s/n. If not given, each step will be taken
        so as to include exactly one more fiber in the annular bin.
    fiber_radius - float, default=2.08
        The radius of each fiber. The default is 2.08, which
        is the arcsec radius of the VIRUS-P fibers.
    s2n_limit float, default=20
        The s/n threshold above which to consider a bin valid. The
        default of 20 is used for MASSIVE folded bins.
    combination - function, default=inverse_variance
        A function to be used to combine fiber spectra. This must take
        the inputs spectra, noise, and mask, in that order, and then
        return a combined spectrum, noise, and mask, also in that
        order, as the elements of an iterable of length 3.

    Returns: binned_fibers, binned_data, annular_sets
    binned_fibers - list
        A list of lists, with a sublist for each bin giving the fiber
        numbers of the fibers included in that bin
    binned_data - list
        A list with an entry for each bin, ordered identical to
        binned_fibers. Each entry is a 2d array giving the bin's
        spectral data: the first row is the normalized spectrum,
        the second the noise estimate, and the third a masking array.
    annular_sets - list
        A list giving the polar boundaries of the bins. Each bin is
        two disjoint areas, and this gives the boundaries for the
        binned region on the northern side of the major axis. The
        second region is the reflection of the given region across
        the major axis. This list contains an entry for each annular
        set of bins, which in turn contains the radial and angular
        boundaries of that set of bins in the format:
        [[rin, rout], [(a0, a1), (a1, a2), (a2, a3), ... ]].
    
    Bin formating conventions:
    For the bin in the nth radial position and the mth angular
    position (n, m starting at 0) on the north side of the major axis,
    that bin is identified as bin number q = n + m and has the
    properties:
      - the fibers in bin q are binned_fibers[q]
      - the spectrum of bin q is binned_data[q][0, :]
      - the noise in the spectrum of bin q is binned_data[q][1, :]
      - the masked wavelength pixels in bin q are binned_data[q][2, :]
      - the radial bounds of the north section of bin q
        are annular_sets[n][0]
      - the angular bounds of the north section of bin q
        are annular_sets[n][1][m]
    """
    radii = np.sqrt(np.sum(coords**2, axis=1))
    angles = np.arctan2(coords[:, 1], coords[:, 0])  # output in (-pi, pi]
    major_axis = principal_value_shift(major_axis, lower=0.0, period=np.pi)
    angles -= major_axis  # go to frame with major axis angle = 0
    angles = principal_value_shift(angles, lower=0.0, period=2*np.pi)
        # angles needed in range (0, 2pi] for comparisons below
    single_fiber_radius = (
        find_single_fiber_region(spectra, noise, mask, radii,
                                 fiber_radius=fiber_radius,
                                 s2n_limit=s2n_limit))
    if fibers_id is None:
        fibers_id = np.arange(spectra.shape[0], dtype=int)
    is_single_fiber = (radii <= single_fiber_radius)
    single_fibers = fibers_id[is_single_fiber]
    single_binned_fibers = [[f] for f in single_fibers]
    [single_normed_spectra,
     single_normed_noise] = median_normalize(spectra[single_fibers, :],
                                             noise=noise[single_fibers, :],
                                             mask=mask[single_fibers, :],
                                             mask_fill=None)
    single_binned_data = map(np.array, zip(single_normed_spectra,
                                           single_normed_noise,
                                           mask[single_fibers, :]))
    remaining = ~is_single_fiber
    multi_binned_fibers, multi_binned_data, annular_sets = (
        construct_polar_multifiber_bins(fibers_id[remaining],
                          spectra[remaining, :],
                          noise[remaining, :], mask[remaining, :],
                          radii[remaining], angles[remaining],
                          starting_radius=single_fiber_radius,
                          s2n_limit=s2n_limit, step_size=step_size,
                          combination=combination,
                          angular_paritioner=angular_paritioner,
                          major_axis=major_axis))


    binned_fibers = single_binned_fibers + multi_binned_fibers
    binned_data = single_binned_data + multi_binned_data
    return binned_fibers, binned_data, annular_sets, single_fiber_radius