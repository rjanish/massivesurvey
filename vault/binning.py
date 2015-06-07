"""
This module performs spacial binning of ifu data and contains
functions for working with previously binned data.
"""

import numpy as np
np.seterr(all='raise')   # force numpy warnings to raise exceptions
np.seterr(under='warn')  # for all but underflow warnings (numpy
                         # version 1.8.2 raises spurious underflows
                         # on some masked array computations)
                        

#################################################################
# For working with Jenny Greene's binned VIRUS-P data:

from itertools import groupby, chain


def parse_bindefs(bin_def_filename, binned_fibers_filename, ma_ang,
                  unbinned_fibers_filename, unbinned_file_present,
                  reflect=False):
    """
    Parse the MASSVIE bin definition files output by Jenny Greene's
    binning code into into a list of fiber numbers, bin boundaries,
    radial bin groupings.

    Args:
    bin_def_filename - string
        The file listing the radial and angular bounds of each
        multi-fiber bin, usually 'CoAddBinSizeInfo.txt'
    binned_fibers_filename - string
        The file listing the fibers contained in each multi-fiber
        bin, usually 'CoAddBinInfo.txt'
    ma_ang - float
        The angle of the galaxy major axis, measured N of E.  This
        is not the conventional position angle!
    unbinned_fibers_filename - string
        The file listing the fibers of the singe-fiber bins,
        usually 'CentreBinInfo.txt'
    unbinned_file_present - bool
        True if there exists a file corresponding to
        unbinned_fibers_filename detailing the single-fiber bins
    reflect - bool
        If True, coordinates will be reflected across North-South.
        This is useful for portraying the binning with E to the left.

    Returns: fibers_in_bin, bin_outlines, radial_sets
    fibers_in_bin - list
        A list with one entry per bin and with each entry consisting
        of a list of all fiber numbers included in that bin
    bin_outlines - list
        A list with one entry per bin and with each entry consisting
        of a shapely polygon object corresponding to the bin's
        boundary for multi-fiber bins, while for single-fiber bins
        the entry is None.
    radial_sets - list
        The bin geometry is polar, with each bin a section of an
        annulus.  This is a list, which each entry detailing the
        division of an annulus into bins.  The format of an entry is:
            [[rin, rout], [(a0, a1), (a1, a2), (a2, a3), ... ]].
        This gives first the radial boundaries of the annulus, and
        then the angular boundaries of each bin in the annulus.

    The ordering of bins is consistent across all three outputs, so
    for the bin in the nth radial section and the mth angular section
    within that radial section (n, m starting at 0), that bin is
    described by bin number q = n + m and has the properties:
        the fibers in bin q are fibers_in_bin[q]
        the boundary polygon of bin q is bin_outlines[q]
        the radial bounds of bin q are radial_sets[n][0]
        the angular bounds of bin q are radial_sets[n][1][m]
    """
    # read bin definitions
    bin_defs = []
        # bin_defs - each element is the entry in "CoAddBinSizeInfo.txt"
        # that specifies a set of bins at a particular radius
    with open(bin_def_filename, 'r') as bin_def_file:
        grouped = groupby(bin_def_file, lambda l: l == '\n')
        for delimiter, lines in grouped:
            lines = list(lines)
            if (not delimiter) and (len(lines) > 1):
                bin_set = [map(float, l.split()) for l in lines]
                bin_defs.append(bin_set)
    bin_defs = sorted(bin_defs, key=lambda l: l[0][0])
    # process bin definitions
    radial_sets = []
        # radial_sets - lists all the bins.
        # for radial bin number n and angular bin number m, with both
        # n and m starting at 0 and increasing outward and counter-
        # clockwise, the boundaries are give by:
        #   rmin, rmax = radial_sets[n][0]    (arcsec)
        #   amin, amax = radial_sets[n][1][m] (degrees)
    for bin_set in bin_defs:
        rin, rout, num_angular = bin_set[0]
        rin, rout = rin, rout
        angular_divider = len(bin_set[1:])/2
        ang_start = list(chain(*bin_set[1:1 + angular_divider]))
        ang_end = list(chain(*bin_set[1 + angular_divider:]))
        ang_start = ma_ang + np.array(ang_start)*(180.0/np.pi)
        ang_end = ma_ang + np.array(ang_end)*(180.0/np.pi)
        if reflect:
            ang_end, ang_start = 2*ma_ang - np.array([ang_start, ang_end])
        radial_sets.append([[rin, rout], zip(ang_start, ang_end)])
    fibers_in_bin = []
        # fibers_in_bin[n] - list of coordinates of all fibers in the nth
        # bin, with coordinates as tuples in Cartesian arcsec
    bin_outlines = []
        # bin_outlie[n] - a shapely polygon giving the boundaries of the
        # nth bin; or None if the bin is a single fiber
    if unbinned_file_present:
        with open(unbinned_fibers_filename, 'r') as unbinned:
            lines = [int(l) for l in unbinned.readlines()
                                  if len(l.split()) > 0]
            num_singles = lines[0] # first line of file - number unbined
            singles = lines[1:]    # all other lines are unbined fibers
            for fiber in singles:
                fibers_in_bin.append([fiber])
                bin_outlines.append(None)
    with open(binned_fibers_filename, 'r') as binned:
        grouped = groupby(binned, lambda l: l == '\n')
        group_counter = 0
        for delimiter, lines in grouped:
            if not delimiter:
                if group_counter > 0:
                    # first group of file is a header, skip it
                    lines = list(lines)
                    rnum, anum, num_fibers = map(int, lines[0].split())
                    fibers = [map(int, l.split()) for l in lines[1:]]
                    fibers = list(chain(*fibers))
                    if fibers == [0]:
                        fibers = []
                        # bins with no fibers are marked as containing
                        # the single fiber zero
                    # subsequent group starts with a line of identifiers,
                    # followed by line(s) listing the fibers in the bin
                    rmin, rmax = radial_sets[rnum][0]
                    amin, amax = radial_sets[rnum][1][anum]
                    bin_poly = polar_box(rmin, rmax, amin, amax)
                    fibers_in_bin.append(fibers)
                    bin_outlines.append(bin_poly)
                group_counter += 1
    return fibers_in_bin, bin_outlines, radial_sets


def compute_bindata(fibers_in_bin, coords, fluxes,
                    flux_errors, ma_ang, folded=False):
    """
    Computes the total flux of each bin and the flux-weighted center
    of each bin.  For bins containing fibers folded across the major
    axis, the center is computed after folding.

    Args:
    fibers_in_bin - list
        A list of the fibers contained in each bin. Fiber numbers
        must start with zero and proceed upwards.
    coords - 2d ndarray
        An array giving the coordinates of each fiber.
    fluxes - 1d ndarray
        An array giving the total flux of each fiber.
    flux_errors - 1d ndarray
        An array giving the estimated error in the total fiber fluxes.
    ma_ang - float
        The angle of the galaxy major axis, measured N of E.  This
        is not the conventional position angle!
    folded - bool or 1d array of bools, default=False
        True if bin contains folded fibers.  If not an array,
        assumed to apply to all bins.

    Returns: weighted_positions, bin_fluxes
    weighted_positions - 2d array
        Weighted center of each bin
    bin_fluxes - 1d array
        Total flux of each bin
    """
    if type(folded) is bool:
        folded = [folded]*len(fibers_in_bin)
    weighted_positions, bin_fluxes, bin_flux_errors = [], [], []
    for fibers, fold in zip(fibers_in_bin, folded):
        fiber_fluxes = fluxes[fibers]
        total_flux = fiber_fluxes.sum()
        x, y = coords[fibers, 0], coords[fibers, 1]
        if (len(fibers) > 1) and fold:
            slope = np.tan(np.deg2rad(ma_ang))
            fibers_below = np.array([x[slope*x > y],
                                     y[slope*x > y]]).T
            fibers_above = np.array([x[slope*x <= y],
                                     y[slope*x <= y]]).T
            major_axis_vector = (
                np.array([1.0, slope])/np.sqrt(1.0 + slope**2))
            dot_product = (fibers_below[:, 0]*major_axis_vector[0] +
                           fibers_below[:, 1]*major_axis_vector[1])
            x_refl = (2*dot_product*major_axis_vector[0] -
                      fibers_below[:, 0])
            y_refl = (2*dot_product*major_axis_vector[1] -
                      fibers_below[:, 1])
            x = np.concatenate((fibers_above[:, 0], x_refl))
            y = np.concatenate((fibers_above[:, 1], y_refl))
        weighted_positions.append([np.sum(x*bin_fluxex)/total_flux,
                                   np.sum(y*bin_fluxex)/total_flux])
        bin_fluxes.append(total_flux)
        bin_flux_errors.append(np.sqrt(np.sum(flux_errors[fibers]**2)))
    weighted_positions = np.array(weighted_positions)
    bin_fluxes = np.array(bin_fluxes)
    bin_flux_errors = np.array(bin_flux_errors)
    return weighted_positions, bin_fluxes, bin_flux_errors


def get_binned_spectra(bin_files, valid_lower=0.0, valid_upper=10**3):
    """
    Read, mask, and normalize binned spectra from the MASSIVE binned
    data files produced by Jenny Greene's binning scripts. Spectra
    are normalized to have median of 1.0, and are masked such that all
    spectra values are within the specified range.  Masked values will
    be replaced by linear interpolation and their positions recorded.

    Args:
    bin_files - list
        List of names of binned data files.  The order in which the
        files are listed will be used in ordering the returned data.
    valid_lower = float, default=0
        Smallest data value considered valid, all spectrum values
        less than valid_lower will be masked.  The default will catch
        the vaccine masking flag of -666.0
    valid_upper = float, default=10**3
        Largest data value considered valid, all spectrum values
        greater than valid_upper will be masked.

    Returns: binned_data, logscales
    binned_data - 3d ndarray
        Data for each bin.  The first axis corresponds to the bins,
        which are arranged by radius in the order of the passed
        bin data files and within each radius in order of increasing
        starting angle.  If the bin data files are sorted to start
        with the innermost and go to the outermost radial section,
        then these are ordered outwards and counter-clockwise.
        For each bin, this contains a 2d array, with wavelengths
        running along the column and each row containing a different
        data type: the first row is the spectrum, the second is the
        noise, the third is the wavelength sample points, and the
        final fourth row is the bad pixel mask (True on bad pixels).
    logscales - 1d ndarray
        For each bin, this gives the logscale of the wavelength
        sampling, where the logscale for log-spaced wavelengths is
        defined as:
            log(wavelengths[n]) = log(wavelengths[0]) + n*logscale
        This is only meaningful for log-spaced sampling.
    """
    # get individual bin data
    binned_data, logscales = [], []
    for f in bin_files:
        hdu = fits.open(f)
        data = hdu[0].data
        hdu.close()
        for bin_subnumber in range(data.shape[1]):
            spectrum = data[0, bin_subnumber, :]
            noise = data[1, bin_subnumber, :]
            mask = ((spectrum < valid_lower) | (noise < valid_lower) |
                    (spectrum > valid_upper) | (noise > valid_upper))
            masked = np.argwhere(mask)
            nonmasked = np.argwhere(~mask)
            for m in masked:  # fill masked values by linear interp
                above = (nonmasked > m).any()
                below = (nonmasked < m).any()
                if above and below:
                    above_index = np.min(nonmasked[nonmasked > m])
                    below_index = np.max(nonmasked[nonmasked < m])
                elif (not above) and below:
                    below_index = np.max(nonmasked[nonmasked < m])
                    above_index = below_index
                elif above and (not below):
                    above_index = np.min(nonmasked[nonmasked > m])
                    below_index = above_index
                else:
                    raise Exception("cannot fill masked value")
                spectrum[m] = (spectrum[below_index] +
                               spectrum[above_index])*0.5
                noise[m] = (noise[below_index] +
                            noise[above_index])*0.5
            norm_value = np.median(spectrum)
            spectrum = spectrum/norm_value
            noise = noise/norm_value
            wavelengths = data[2, bin_subnumber, :]
            logscale = np.log(wavelengths[1]/wavelengths[0])
                # assumes wavelength are log-spaced, true for MASSIVE
            binned_data.append([spectrum, noise, wavelengths, mask])
            logscales.append(logscale)
    return np.array(binned_data), np.array(logscales)


def bin_union(binned_data, bin_fluxes):
    """
    Combined the given binned data to get the data for a bin
    which is the union of all the passed bins.  The data are
    combined with a luminosity weighted average.

    Args:
    binned_data - 3d ndarray
        The bin data, in the format of get_binned_spectra, for all
        bins that are to be combined.  It is assumed that all bins
        have normalized spectra and an identical wavelength sampling.
    binned_data - 1d ndarray
        The fluxes of the bins to be combined, ordered in to be in
        correspondence with the first axis of binned_data

    Return: full_data, logscale
    full_data - 2d array
        The bin data, in the formated of get_binned_spectra, for
        the combined bin
    logscale - float
        The logscale, as defined in get_binned_spectra, for the
        combined bin
    """
    normed_spectrum = binned_data[:, 0, :]
    normed_noise = binned_data[:, 1, :]
    wavelengths = binned_data[:, 2, :]
    mask = binned_data[:, 3, :]
    unnormed_spectrum = ((normed_spectrum.T)*bin_fluxes).T
    unnormed_noise = ((normed_noise.T)*bin_fluxes).T
    coadd_spectrum = np.sum(unnormed_spectrum, axis=0)
    coadd_noise = np.sqrt(np.sum(unnormed_noise**2, axis=0))
        # assumes an identical wavelength sampling
    full_norm_value = np.median(coadd_spectrum)
    normed_coadd_spectrum = full_coadd/full_norm_value
    normed_coadd_noise = full_noise/full_norm_value
    coadd_mask = np.sum(unnormed_mask, axis=0) > 0
        # equivalent to an or over bins by wavelength
        # assumes an identical wavelength sampling
    full_data = [full_coadd, full_noise, wavelengths, full_mask]
    logscale = np.log(wavelengths[0, 1]/wavelengths[0, 0])
        # assumes an identical wavelength sampling
    return np.array(full_data), logscale


#################################################################
# For binning data from fiber spectra and coordinates:

from util import principal_value_shift


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


def combine_spectra_flux(spectra, noise, mask=None, mask_fill=None):
    """
    Combine multiple spectra with noise and possible masking,
    using a flux weighted mean at each wavelength.

    Args:
    spectrum - 2d float array
        An array of spectra to combine, with each row a spectrum
        and each spectrum identically sampled in wavelength. No
        normalization is assumed.
    noise - 2d float array
        The estimated standard deviation of the noise in spectrum.
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
    normed_spectra, normed_noise = median_normalize(spectra, noise, mask)
    flux = np.ma.sum(np.ma.array(spectra, mask=mask), axis=1).data
    relative_weights = np.vstack((flux,)*spectra.shape[1]).T  # each column is identical
    relative_weights[mask] = 0.0  # removes masked pixels from weighted mean
    total_weight = relative_weights.sum(axis=0)
    all_masked = np.all(mask, axis=0)
    total_weight[all_masked] = 1.0
        # set nonzero fiducial total weight of wavelengths with no un-masked
        # values to avoid division errors; normalized weight is still zero
    weights = relative_weights/total_weight  # divides each col by const
    [mean_spectrum, mean_noise,
     mean_mask, clipped] = clipped_mean(normed_spectra, weights=weights,
                                        noise=normed_noise, mask=mask)
    mean_spectrum, mean_noise = median_normalize(mean_spectrum, mean_noise,
                                                 mean_mask, mask_fill)
        # median_normalize will either float or
        # fill masked values according to mask_fill
    return mean_spectrum, mean_noise, mean_mask


def combine_spectra_ivar(spectra, noise, mask=None, mask_fill=None):
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


def find_single_fiber_boundary(spectra, noise, radius, mask=None,
                               fiber_radius=2.08, s2n_limit=20/np.sqrt(2)):
    """
    Compute the maximum radius below which all fibers are above a s/n
    threshold. To avoid future rounding issues, the radius is not set
    to pass through a fiber center but is forced to be between fibers.
    If no fibers are above the threshold, the returned radius is zero,
    and if all fibers are above the threshold, the returned radius is
    one fiber radius greater than the outermost fiber center.

    Args:
    spectra - 2D arraylike
        The fiber spectra, with each row an individual fiber spectrum.
        All spectra as assumed to be identically sampled.
    noise - 2D arraylike
        The estimated noise in the fiber spectra
    radius - 2D arraylike
        The radial position of each fiber
    mask - 2D boolean arraylike, default=None
        An indicator of bad pixels: where True, the corresponding data
        in spectrum and noise will be ignored when computing the s/n.
        If not passed, all pixels are assumed to be good.
    fiber_radius - float, default=2.08
        The intrinsic radius of each fiber, used to be sure all fibers
        are included in the comparison. The default is 2.08, which
        is the arcsec radius of the VIRUS-P fibers.
    s2n_limit - float, default=20/np.sqrt(2)
        The s/n threshold above which a spectrum is considered good.
        Default is 20/sqrt(2), which gives bins of s/n ~ 20 after
        a single folding, which is the value used for MASSIVE bins.

    Returns: single_fiber_boundary
    single_fiber_boundary - float
        Boundary below which all fibers pass the threshold.
    """
    if mask is None:
        mask = np.zeros(spectra.shape, dtype=bool)  # assume all pixels good
    masked_spectra = np.ma.array(spectra, mask=mask)
    masked_noise = np.ma.array(noise, mask=mask)
    s2n = np.ma.mean(masked_spectra/masked_noise, axis=1).data
    s2n_is_high = (s2n > s2n_limit)
    # set radius below which fibers will be left unbinned,
    # displacing radius from fiber centers to avoid rounding
    # issues when determining the inclusion of fiber.
    min_fiber_high = s2n_is_high[np.argmin(radius)]
    if not min_fiber_high:
        single_fiber_boundary = 0.0
    elif np.all(s2n_is_high):
        single_fiber_boundary = np.max(radius) + fiber_radius
    else:
        supremum = radius[~s2n_is_high].min()
            # maximum radius below which no fibers require binning
        infimum = radius[radius < supremum].max()
            # minimum radius below which no fibers require binning
        single_fiber_boundary = 0.5*(supremum + infimum)
    return single_fiber_boundary


def find_polar_bounds(fibers, spectra, noise, mask, radius, angle,
                      major_axis=0, aspect_ratio=1.5, step_size=None,
                      starting_radius=0.0, s2n_limit=20/np.sqrt(2),
                      combination=combine_spectra_ivar):
    """
    Divides a set of ifu fibers into a polar grid of bins with parity
    across the major and minor axes, ensuring that each bin passes a
    s/n threshold. Bins are created from the center outwards, with any
    excess fibers that cannot be binned below threshold discarded.

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
    aspect_ratio - float, default=1.5
        The aspect ratio of allowed bins, as angular_size/radial_size,
        which will be use to determine how many angular divisions
        are allowed inside one annular division.
    step_size - float, defalt=None
        The radial step size to take when increasing the radius of
        a bin due to low s/n. If not given, each step will be taken
        so as to include exactly one more fiber in the annular bin.
    starting_radius - float, default=0.0
        Radius at which to start the binning.
    s2n_limit float, default=20/np.sqrt(2)
        The s/n threshold above which to consider a bin valid. The
        default is 20/sqrt(2), which gives bins of s/n ~ 20 after a
        single folding, which is the value used for MASSIVE bins.
    combination - function, default=combine_spectra_ivar
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
        A list giving the polar boundaries of the bins. This contains
        an entry for each annular set of bins, which in turn contains
        the radial and angular boundaries of that set of bins in the
        format: [[rin, rout], [(a0, a1), (a1, a2), (a2, a3), ... ]].
    
    Bin formating conventions:
    For the bin in the nth radial position and the mth angular
    position (n, m starting at 0), that bin is identified as bin
    number q = n + m and has the properties:
        the fibers in bin q are binned_fibers[q]
        the spectrum of bin q is binned_data[q][0, :]
        the noise in the spectrum of bin q is binned_data[q][1, :]
        the masked wavelength pixels in bin q are binned_data[q][2, :]
        the radial bounds of bin q are annular_sets[n][0]
        the angular bounds of bin q are annular_sets[n][1][m]
    """
    major_axis = principal_value_shift(major_axis, lower=0.0, period=np.pi)
    angle -= major_axis  # go to frame with major axis angle = 0
    angle = principal_value_shift(angle, lower=0.0, period=2*np.pi)
    # set radial partition
    if step_size is None:
        sorted_radius = np.sort(radius)
        midpoints = (sorted_radius[:-1] + sorted_radius[1:])*0.5
        above_starting = midpoints[midpoints > starting_radius].tolist()
        typical_gap = np.median(midpoints[1:] - midpoints[:-1])
        final_radius = radius.max() + typical_gap
        radial_partition = ([starting_radius] + above_starting +
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
            # make angular partition - angles in frame major_axis_angle = 0
            delta_r = outer_radius - lower_radius
            mid_r = 0.5*(outer_radius + lower_radius)
            target_bin_arclength = delta_r*aspect_ratio
            available_arclength = 0.5*np.pi*mid_r  # one quadrant
            num_bins = int(available_arclength/target_bin_arclength)
            angular_bounds_ne = np.linspace(0.0, 0.5*np.pi, num_bins + 1)
                # angular boundaries in the north east quadrant,
                # including boundaries at 0 and pi/2
            angular_bounds_nw = np.pi - angular_bounds_ne[-2::-1]
                # angular boundaries reflected into the north west quadrant,
                # not including boundary at pi/2, but including pi
            angular_bounds_n = np.concatenate((angular_bounds_ne,
                                               angular_bounds_nw))
                # angular boundaries on the north side of the major axis,
                # including boundaries at 0 and pi
            angular_bounds_s = 2*np.pi - angular_bounds_n[-2::-1]
                # angular boundaries reflected onto the side side of the
                # major axis, not including boundary at pi, including 2*pi
            angular_bounds = np.concatenate((angular_bounds_n,
                                             angular_bounds_s))
            # check s/n of each angular section
            trial_binned_fibers = []
            trial_binned_data = []
            angle_iteration = zip(angular_bounds[:-1], angular_bounds[1:])
            for start_angle, stop_angle in angle_iteration:
                angle_selector = ((start_angle <= angle) &
                                  (angle < stop_angle))
                trial_selector = fibers_in_annulus & angle_selector
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
                sky_angular_bounds = list(angular_bounds + major_axis)
                    # angular boundaries in the frame with E = 0 and N = 90
                sky_angular_pairs = zip(sky_angular_bounds[:-1],
                                        sky_angular_bounds[1:])
                annular_sets.append([radial_bounds, sky_angular_pairs])
                num_radial_steps = outer_iter + 1
                starting_index += num_radial_steps
                break  # start new annular bin
        else:
            # final annulus does not meet binning criteria
            # handle outer fibers.
            break
    return binned_fibers, binned_data, annular_sets


def determine_polar_bins(spectra, noise, mask, coords, major_axis=0,
                         fiber_radius=2.08, aspect_ratio=1.5,
                         s2n_limit=20/np.sqrt(2), step_size=None,
                         combination=combine_spectra_ivar):
    """
    Performs a full polar of binning of the passed spectra, with
    inner, high s/n fibers sorted into single-fiber bins and outer,
    low s/n fibers sorted into polar bins with parity across the
    major and minor axes.

    
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
    coords - 2d array
        2d array with each row corresponding to a fiber, listed in the
        same order as the passed fibers array. Each row contains two
        elements which give the fiber's Cartesian position coordinates
    major_axis - float, default=0
        The angle to the major axis measured from E to N in radians.
        This is not the conventional position angle, which is related
        by: major_axis = pi/2 - position_angle
    aspect_ratio - float, default=1.5
        The aspect ratio of allowed bins, as angular_size/radial_size,
        which will be use to determine how many angular divisions
        are allowed inside one annular division.
    step_size - float, defalt=None
        The radial step size to take when increasing the radius of
        a bin due to low s/n. If not given, each step will be taken
        so as to include exactly one more fiber in the annular bin.
    fiber_radius - float, default=2.08
        The intrinsic radius of each fiber, used to be sure all fibers
        are included in the comparison. The default is 2.08, which
        is the arcsec radius of the VIRUS-P fibers.
    s2n_limit float, default=20/np.sqrt(2)
        The s/n threshold above which to consider a bin valid. The
        default is 20/sqrt(2), which gives bins of s/n ~ 20 after a
        single folding, which is the value used for MASSIVE bins.
    combination - function, default=combine_spectra_ivar
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
        A list giving the polar boundaries of the bins. This contains
        an entry for each annular set of bins, which in turn contains
        the radial and angular boundaries of that set of bins in the
        format: [[rin, rout], [(a0, a1), (a1, a2), (a2, a3), ... ]].
    
    Bin formating conventions:
    For the bin in the nth radial position and the mth angular
    position (n, m starting at 0), that bin is identified as bin
    number q = n + m and has the properties:
        the fibers in bin q are binned_fibers[q]
        the spectrum of bin q is binned_data[q][0, :]
        the noise in the spectrum of bin q is binned_data[q][1, :]
        the masked wavelength pixels in bin q are binned_data[q][2, :]
        the radial bounds of bin q are annular_sets[n][0]
        the angular bounds of bin q are annular_sets[n][1][m]
    """
    radius = np.sqrt(np.sum(coords**2, axis=1))
    angle = np.arctan2(coords[:, 1], coords[:, 0])  # output in (-pi, pi)
    angle = principal_value_shift(angle, lower=0.0, period=2*np.pi)
    single_fiber_boundary = (
        find_single_fiber_boundary(spectra, noise, radius, mask=mask,
                                   fiber_radius=fiber_radius,
                                   s2n_limit=s2n_limit))
    fibers = np.arange(spectra.shape[0], dtype=int)
    is_single_fiber = radius <= single_fiber_boundary
    single_fibers = fibers[is_single_fiber]
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
        find_polar_bounds(fibers[remaining], spectra[remaining, :],
                          noise[remaining, :], mask[remaining, :],
                          radius[remaining], angle[remaining],
                          major_axis=major_axis, aspect_ratio=aspect_ratio,
                          starting_radius=single_fiber_boundary,
                          s2n_limit=s2n_limit, step_size=step_size,
                          combination=combination))
    binned_fibers = single_binned_fibers + multi_binned_fibers
    binned_data = single_binned_data + multi_binned_data
    return binned_fibers, binned_data, annular_sets


def find_polar_bounds_folded(fibers, spectra, noise, mask, radius, angle,
                             major_axis=0, aspect_ratio=1.5, step_size=None,
                             starting_radius=0.0, s2n_limit=20,
                             combination=combine_spectra_ivar):
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
    aspect_ratio - float, default=1.5
        The aspect ratio of allowed bins, as angular_size/radial_size,
        which will be use to determine how many angular divisions
        are allowed inside one annular division.
    step_size - float, defalt=None
        The radial step size to take when increasing the radius of
        a bin due to low s/n. If not given, each step will be taken
        so as to include exactly one more fiber in the annular bin.
    starting_radius - float, default=0.0
        Radius at which to start the binning.
    s2n_limit float, default=20
        The s/n threshold above which to consider a bin valid. The
        default is 20, which is used for MASSIVE folded bins.
    combination - function, default=combine_spectra_ivar
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
    major_axis = principal_value_shift(major_axis, lower=0.0, period=np.pi)
    angle -= major_axis  # go to frame with major axis angle = 0
    angle = principal_value_shift(angle, lower=0.0, period=2*np.pi)
    # set radial partition
    if step_size is None:
        sorted_radius = np.sort(radius)
        midpoints = (sorted_radius[:-1] + sorted_radius[1:])*0.5
        above_starting = midpoints[midpoints > starting_radius].tolist()
        typical_gap = np.median(midpoints[1:] - midpoints[:-1])
        final_radius = radius.max() + typical_gap
        radial_partition = ([starting_radius] + above_starting +
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
            # make angular partition - angles in frame major_axis_angle = 0
            delta_r = outer_radius - lower_radius
            mid_r = 0.5*(outer_radius + lower_radius)
            target_bin_arclength = delta_r*aspect_ratio
            available_arclength = 0.5*np.pi*mid_r  # one quadrant
            num_firstquad = int(available_arclength/target_bin_arclength)
            num_bins_north = 2*num_firstquad
            angular_bounds_n = np.linspace(0.0, np.pi, num_bins_north + 1)
                # angular boundaries on the north side of the major axis,
                # including boundaries at 0 and pi

            # check s/n of each angular section
            trial_binned_fibers = []
            trial_binned_data = []
            angle_iteration = zip(angular_bounds_n[:-1],
                                  angular_bounds_n[1:])
            for start_angle, stop_angle in angle_iteration:
                angle_selector_n = ((start_angle <= angle) &
                                    (angle < stop_angle))
                reflected_start = 2*np.pi - stop_angle
                reflected_stop = 2*np.pi - start_angle
                angle_selector_s = ((reflected_start <= angle) &
                                    (angle < reflected_stop))
                angle_selector = angle_selector_n | angle_selector_s
                trial_selector = angle_selector & fibers_in_annulus
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
                sky_angular_bounds_n = list(angular_bounds_n + major_axis)
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


def determine_polar_bins_folded(spectra, noise, mask, coords, major_axis=0,
                                fiber_radius=2.08, aspect_ratio=1.5,
                                s2n_limit=20, step_size=None,
                                combination=combine_spectra_ivar):
    """
    Performs a full polar of binning of the passed spectra, with
    inner, high s/n fibers sorted into single-fiber bins and outer,
    low s/n fibers sorted into polar bins with parity across the
    major and minor axes.

    
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
    coords - 2d array
        2d array with each row corresponding to a fiber, listed in the
        same order as the passed fibers array. Each row contains two
        elements which give the fiber's Cartesian position coordinates
    major_axis - float, default=0
        The angle to the major axis measured from E to N in radians.
        This is not the conventional position angle, which is related
        by: major_axis = pi/2 - position_angle
    aspect_ratio - float, default=1.5
        The aspect ratio of allowed bins, as angular_size/radial_size,
        which will be use to determine how many angular divisions
        are allowed inside one annular division.
    step_size - float, defalt=None
        The radial step size to take when increasing the radius of
        a bin due to low s/n. If not given, each step will be taken
        so as to include exactly one more fiber in the annular bin.
    fiber_radius - float, default=2.08
        The intrinsic radius of each fiber, used to be sure all fibers
        are included in the comparison. The default is 2.08, which
        is the arcsec radius of the VIRUS-P fibers.
    s2n_limit float, default=20
        The s/n threshold above which to consider a bin valid. The
        default is 20, which is used for MASSIVE folded bins.
    combination - function, default=combine_spectra_ivar
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
    radius = np.sqrt(np.sum(coords**2, axis=1))
    angle = np.arctan2(coords[:, 1], coords[:, 0])  # output in (-pi, pi)
    angle = principal_value_shift(angle, lower=0.0, period=2*np.pi)
    single_fiber_boundary = (
        find_single_fiber_boundary(spectra, noise, radius, mask=mask,
                                   fiber_radius=fiber_radius,
                                   s2n_limit=s2n_limit))
    fibers = np.arange(spectra.shape[0], dtype=int)
    is_single_fiber = radius <= single_fiber_boundary
    single_fibers = fibers[is_single_fiber]
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
        find_polar_bounds_folded(fibers[remaining], spectra[remaining, :],
                          noise[remaining, :], mask[remaining, :],
                          radius[remaining], angle[remaining],
                          major_axis=major_axis, aspect_ratio=aspect_ratio,
                          starting_radius=single_fiber_boundary,
                          s2n_limit=s2n_limit, step_size=step_size,
                          combination=combination))
    binned_fibers = single_binned_fibers + multi_binned_fibers
    binned_data = single_binned_data + multi_binned_data
    return binned_fibers, binned_data, annular_sets