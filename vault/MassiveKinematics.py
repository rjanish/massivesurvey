"""
Script to analyze stellar kinematics of galaxies bin-by-bin with pPXF.
From binned spectra with position and resolution information this computes
the best-fitting Gauss-Hermite kinematic parameters in each bin.
"""


import sys  # needed for driver section
import os   # needed for driver section
import pickle   # needed for driver section
sys.path.append("/stg/scripts/bin/ppxf")
sys.path.append("/stg/scripts/bin/mpfit")
sys.path.append("../massive")
import re
from itertools import groupby, chain

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle
from descartes import PolygonPatch
from astropy.io import fits

from plotting.geo_utils import polar_box
from ppxf import ppxf

#import binning as make_bins


speed_of_light = 299792.458 # km/s


def gaussian(x, central, sigma):
    """ Returns the normalized element-wise Gaussian G(x) for an array x. """
    arg = (x - central)/sigma
    norm = sigma*np.sqrt(2*np.pi)
    return np.exp(-0.5*(arg**2))/norm


def read_dict_file(filename, conversions, comment='#'):
    """
    Parse the passed file into dictionary.

    Args:
    filename - string
        Name of file to be parsed. The file should contain on each
        line first a string that will become the key name, and then
        the corresponding value separated by whitespace. This must be
        the only whitespace on the line.  Leading the trailing
        whitespace will be trimmed.  If a key appears more than once,
        all values will be put into a list in order of appearance.
    comment - string, default='#'
       Data appearing after a comment character on line is ignored
    conversion - list
        A list of tuples of the form (keys, func), where keys is a list
        of keys used in the target file and func is a conversion
        function that will be applied to the corresponding values entries
        in the target file before being stored in the dictionary.

    Returns - file_dict
    file_dict - dictionary
    """
    conversions = sorted(conversions, key=lambda l: len(l[0]))
        # sort by number of targeted fields, nominally faster later
    info = {}
    num_encountered = {}
    with open(filename, 'r') as infofile:
        for line in infofile:
            comment_location = line.find(comment)
            if comment_location != -1:  # find gives -1 if nothing is found
                line = line[:comment_location]
            field_name, field_value = line.split()
            for category, convert_func in conversions:
                if field_name in category:
                    if field_name in num_encountered:
                        num_encountered[field_name] += 1
                    else:
                        num_encountered[field_name] = 1
                    if num_encountered[field_name] == 1:
                        info[field_name] = convert_func(field_value)
                    elif num_encountered[field_name] == 2:
                        info[field_name] = [info[field_name]]
                        info[field_name].append(convert_func(field_value))
                    elif num_encountered[field_name] >= 3:
                        info[field_name].append(convert_func(field_value))
                    break  # once conversion func located, stop searching
    return info


def read_target_info(filename):
    """
    Parse the target info file into dictionary. See readme.txt for
    the assumed formating of the target info file.
    """
    as_string = ["name", "fiber_data", "data_dir","ir_file"]
    as_float = ["center_ra", "center_dec", "pa", "fiber_radius"]
    as_int = ["first_fiber", "last_fiber","rm_fiber"]
    conversions = [(as_string, str), (as_int, int), (as_float, float)]
        # for regular expression pattern, when read from a file any control
        # characters are automatically escaped, so converting straight to
        # string is fine, no raw string nonsense is needed
    return read_dict_file(filename, conversions)


def read_fitting_info(filename):
    """
    Parse the target info file into dictionary. See readme.txt for
    the assumed formating of the target info file.
    """
    as_string = ["results_dir", "intermediates_dir", "template_dir",
                 "template_index","fit_by"]
    as_float = ["template_fwhm", "bias", "v_guess", "sigma_guess", "hn_guess"]
    as_int = ["additive_degree", "multiplicative_degree", "moments_to_fit"]
    as_range = ["mask_observed", "mask_rest"]
        # format is ##-##, convert to [##, ##] pair representing an interval
    conversions = [(as_string, str), (as_int, int), (as_float, float),
                   (as_range, lambda s: map(float, s.split('-')))]
    return read_dict_file(filename, conversions)


def re_filesearch(pattern, directory=None):
    """
    Return a list of all files in passed directory that match the
    passed regular expression.  Only the filename, not the full
    pathnames are considering for matching, however the returned
    matches will include pathnames if a directory is specified.

    Args:
    pattern - string, regular expression pattern
    dir - string, directory name, default is current directory

    Returns: files
    files - list of strings, matching file paths including directory.
    """
    if directory is None:
        directory = os.curdir
    files_present = os.listdir(directory)
    matches = ["{}/{}".format(directory, f)
               for f in files_present if re.match(pattern, f)]
    return matches


def parse_bindefs(bin_def_filename, binned_fibers_filename, ma_ang,
                  unbinned_fibers_filename, unbinned_file_present):
    """
    Parse the MASSVIE "CoAddBinSizeInfo.txt", "CoAddBinInfo.txt", and
    "CentreBinInfo.txt" binning definition files into a list of fiber
    numbers per bin and a list of polygon boundaries per mulit-fiber bin.
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
                    # subsequent group starts with a line of identifiers,
                    # followed by line(s) listing the fibers in the bin
                    rmin, rmax = radial_sets[rnum][0]
                    amin, amax = radial_sets[rnum][1][anum]
                    bin_poly = polar_box(rmin, rmax, amin, amax)
                    fibers_in_bin.append(fibers)
                    bin_outlines.append(bin_poly)
                group_counter += 1
    return fibers_in_bin, bin_outlines, radial_sets


def parse_fiber_data(master_fits, center):
    """
    Parse the MASSVIE VIRUS-P individual fiber data file, typically
    'RnovallfibNGC????_log.fits', into fiber coordinates and fluxes.
    """
    # get fiber coordinates, flux
    master = fits.open(master_fits)
    coords = master[3].data  # col 0: ra, col 1: dec, in degrees
    spectra = master[0].data # erg/sec/cm^2/angstrom
    errors = master[1].data  # erg/sec/cm^2/angstrom
    waves = master[2].data   # angstrom
    waves = np.vstack((waves,)*3)
        # Wavelengths are initially gives for only one dither, while spectra
        # and coords are given for all three dithers. Wavelengths are
        # identical for all fibers and all dithers, so this matches shapes.
    master.close()
    coords = np.array([coords[:, 0] - center[0],
                       coords[:, 1] - center[1]]).T*3600
    coords[:, 0] = coords[:,0]*np.cos(np.deg2rad(center[1]))
        # coords = [ra wrt gal center, dec wrt gal center] in arcsec
    mask = ((spectra < 0) | (spectra > 10**3))  # bad sky subtraction, cr's
    spectra = np.ma.masked_array(spectra, mask=mask)
    errors = np.ma.masked_array(errors, mask=mask)
    waves = np.ma.masked_array(waves, mask=mask)
    # compute total flux via trapezoid integration, in erg/sec/cm^2
    disp = waves[:, 1:] - waves[:, :-1]
    midpoint_height = (spectra[:, 1:] + spectra[:, :-1])*0.5
    flux = np.sum(disp*midpoint_height, axis=1)
    # ADD FLUX NOISE ESTIMATE
    flux_noise = None
    return coords, flux, flux_noise


def compute_bindata(fibers_in_bin, coords, fluxes, ma_ang):
    """
    Computes the total flux of each bin, as well as the flux-weighted
    center of each bin, taking into account the fact that the bins
    are defined with reflection: the weighted average return is obtained
    by first reflecting all fibers in the bin to be on the same side
    of the major axis.
    """
    weighted_position, bin_flux = [], []
    for fibers in fibers_in_bin:
        bin_fluxex = fluxes[fibers]
        total_flux = bin_fluxex.sum()
        x, y = coords[fibers, 0], coords[fibers, 1]
        weighted_position.append([np.sum(x*bin_fluxex)/total_flux,
                                  np.sum(y*bin_fluxex)/total_flux])
        bin_flux.append(total_flux)
    weighted_position = np.array(weighted_position)
    bin_flux = np.array(bin_flux)
    return weighted_position, bin_flux

def compute_bindata_folded(fibers_in_bin, coords, fluxes, ma_ang):
    weighted_position, bin_flux = [], []
    for fibers in fibers_in_bin:
        bin_fluxex = fluxes[fibers]
        total_flux = bin_fluxex.sum()
        x, y = coords[fibers, 0], coords[fibers, 1]
        if len(fibers) > 1:
            slope = np.tan(np.deg2rad(ma_ang))
            fibers_below = np.array([x[slope*x > y], y[slope*x > y]]).T
            fibers_above = np.array([x[slope*x <= y], y[slope*x <= y]]).T
            major_axis_vector = np.array([1.0, slope])/np.sqrt(1.0 + slope**2)
            dot_product = (fibers_below[:, 0]*major_axis_vector[0] +
                           fibers_below[:, 1]*major_axis_vector[1])
            x_refl = 2*dot_product*major_axis_vector[0] - fibers_below[:, 0]
            y_refl = 2*dot_product*major_axis_vector[1] - fibers_below[:, 1]
            x = np.concatenate((fibers_above[:, 0], x_refl))
            y = np.concatenate((fibers_above[:, 1], y_refl))
        weighted_position.append([np.sum(x*bin_fluxex)/total_flux,
                                  np.sum(y*bin_fluxex)/total_flux])
        bin_flux.append(total_flux)
    weighted_position = np.array(weighted_position)
    bin_flux = np.array(bin_flux)
    return weighted_position, bin_flux


def get_binned_spectra(bin_files):
    """
    Read, mask, and normalize all binned spectra from the MASSVIE bin
    files which have logspaced wavelength sampling, also create a
    full-galaxy bin, assigned bin number 0.  Other bins are assigned
    numbers increasing radially outward and counter-clockwise from
    the major axis, with single-fiber bins first.
    """
    # get individual bin data
    unnormed_spectra, unnormed_noise, unnormed_mask = [], [], []
    bin_counter = 1  # start bin numbers at 1, 0 will be full galaxy
    binned_data, logscales = [], []
    for f in bin_files:
        hdu = fits.open(f)
        data = hdu[0].data
        hdu.close()
        for bin_subnumber in range(data.shape[1]):
            spectrum = data[0, bin_subnumber, :]
            noise = data[1, bin_subnumber, :]
            mask = ((spectrum < 0) | (noise < 0) |
                    (spectrum > 10**3) | (noise > 10**3))
            masked = np.argwhere(mask)
            nonmasked = np.argwhere(~mask)
            for m in masked:
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
            unnormed_spectra.append(spectrum)
            unnormed_noise.append(noise)
            unnormed_mask.append(mask)
            norm_value = np.median(spectrum)
            spectrum = spectrum/norm_value
            noise = noise/norm_value
            wavelengths = data[2, bin_subnumber, :]
            logscale = np.log(wavelengths[1]/wavelengths[0])
            binned_data.append([spectrum, noise, wavelengths, mask])
            logscales.append(logscale)
            bin_counter += 1
    # compute full galaxy bin
        # this assumes that each bin is sampled identically in wavelength!
    full_coadd = np.sum(unnormed_spectra, axis=0)
    full_noise = np.sqrt(np.sum(np.array(unnormed_noise)**2, axis=0))
    full_mask = np.sum(unnormed_mask, axis=0) > 0
    full_norm_value = np.median(full_coadd)
    full_coadd = full_coadd/full_norm_value
    full_noise = full_noise/full_norm_value
    logscale = np.log(wavelengths[1]/wavelengths[0])
    full_data = [full_coadd, full_noise, wavelengths, full_mask]
    logscales = [logscale] + logscales
    binned_data = [full_data] + binned_data
    return np.array(binned_data), logscales


def prepare_templates(spectra, wavelengths, current_fwhm, target_fwhm,
                      target_deltalog, target_range, crop_factor=5):
    """
    Prepare a set of template spectra for use by pPXF.  Input templates are
    assumed to be sampled at identical wavelengths with identical spectral
    resolution.  Outputs spectra will be smoothed to a desired resolution,
    re-sampled on a log-spaced grid, re-binned to give the log-specific
    flux and normalized to have numerical values near 1.

    Args:
    spectra - 2D arraylike
        Array of template spectra, with spectra occupying the rows,
        units of flux/length where length matches wavelength
    wavelengths - 1D arraylike
        Wavelength sample points of spectra, units of length
    current_fwhm - float
        FWHM resolution of templates, units of matching wavelength
    target_fwhm - 1D arraylike
        Desired FWHM resolution of final template spectra, as a function
        of wavelength, units of matching wavelength
    target_deltalog - float
        Desired uniform step in log(wavelength) of final template sampling
    target_range - 1D arraylike
        Desired wavelength range of final template, as [min, max] array,
        units matching wavelengths

    Returns: finished_spectra, logspaced_waves
    finished_spectra - 2D arraylike
        Array of prepared spectra, with spectra occupying the rows,
        units of flux/log(length) where length matches wavelength
    logspaced_waves - 1D arraylike
        Wavelength sampling points of prepared spectra
    """
    # check inputs
    spectra, wavelengths, target_range, target_fwhm = (
        map(np.array, [spectra, wavelengths, target_range, target_fwhm]))
    current_fwhm, target_deltalog = (
        map(float, [current_fwhm, target_deltalog]))
    # compute new wavelength sampling
    slop = 10**(-10)  # prevent accidental extrapolation
    log_range = np.log(target_range)
    log_waves = np.arange(log_range[0]*(1.0 + slop),
                          log_range[1]*(1.0 - slop),
                          target_deltalog)
    logspaced_waves = np.exp(log_waves)
    # compute convolution weights
    fwhm_to_add = np.sqrt(target_fwhm**2 - current_fwhm**2)
    sigma_to_add = fwhm_to_add/(2*np.sqrt(2*np.log(2)))
    measure = wavelengths[1:] - wavelengths[:-1]
    weights = []
    for central_wavelength, sigma in zip(wavelengths, sigma_to_add):
        trapz_samples = np.array([wavelengths[:-1], wavelengths[1:]])
            # sample points needed for trapezoid integration
        psf = gaussian(trapz_samples, central_wavelength, sigma)
        weights.append(psf*measure*0.5)
    weights = np.swapaxes(weights, 0, 1)
    # apply convolution, log transform, re-sampling, and normalization
    finished_spectra = []
    for counter, spectrum in enumerate(spectra):
        smoothed_spectrum = (weights[0, ...]*spectrum[:-1] +
                             weights[1, ...]*spectrum[1:]).sum(axis=1)
            # trapezoid integration of continuous convolution integral
        interpolator = interp1d(wavelengths, smoothed_spectrum)
        interpolated_spectrum = interpolator(logspaced_waves)
        edge_sigmas = target_fwhm[[0, -1]]/(2*np.sqrt(2*np.log(2)))
        edge_buffer = crop_factor*edge_sigmas
        valid = ((wavelengths.min() + edge_buffer[0] < logspaced_waves) &
                 (logspaced_waves < wavelengths.max() - edge_buffer[1]))
        interpolated_spectrum = interpolated_spectrum[valid]
        logspaced_waves = logspaced_waves[valid]
            # crop to eliminate edge effects
        norm = np.median(interpolated_spectrum)
        finished_spectra.append(interpolated_spectrum/norm)
    return np.array(finished_spectra), logspaced_waves
 

def prepare_templates_empirical(spectra, wavelengths, target_fwhm,
                      target_deltalog, target_range, crop_factor=5):
    """
    ALL FALSE:

    Prepare a set of template spectra for use by pPXF.  Input templates are
    assumed to be sampled at identical wavelengths with identical spectral
    resolution.  Outputs spectra will be smoothed to a desired resolution,
    re-sampled on a log-spaced grid, re-binned to give the log-specific
    flux and normalized to have numerical values near 1.

    Args:
    spectra - 2D arraylike
        Array of template spectra, with spectra occupying the rows,
        units of flux/length where length matches wavelength
    wavelengths - 1D arraylike
        Wavelength sample points of spectra, units of length
    current_fwhm - float
        FWHM resolution of templates, units of matching wavelength
    target_fwhm - 1D arraylike
        Desired FWHM resolution of final template spectra, as a function
        of wavelength, units of matching wavelength
    target_deltalog - float
        Desired uniform step in log(wavelength) of final template sampling
    target_range - 1D arraylike
        Desired wavelength range of final template, as [min, max] array,
        units matching wavelengths

    Returns: finished_spectra, logspaced_waves
    finished_spectra - 2D arraylike
        Array of prepared spectra, with spectra occupying the rows,
        units of flux/log(length) where length matches wavelength
    logspaced_waves - 1D arraylike
        Wavelength sampling points of prepared spectra
    """
    # check inputs
    spectra, wavelengths, target_range, target_fwhm = (
        map(np.array, [spectra, wavelengths, target_range, target_fwhm]))
    target_deltalog = float(target_deltalog)
    # compute new wavelength sampling
    slop = 10**(-10)  # prevent accidental extrapolation
    log_range = np.log(target_range)
    log_waves = np.arange(log_range[0]*(1.0 + slop),
                          log_range[1]*(1.0 - slop),
                          target_deltalog)
    logspaced_waves = np.exp(log_waves)
    # compute convolution weights
    # fwhm_to_add = np.sqrt(target_fwhm**2 - current_fwhm**2)
    # sigma_to_add = fwhm_to_add/(2*np.sqrt(2*np.log(2)))
    # measure = wavelengths[1:] - wavelengths[:-1]
    # weights = []
    # for central_wavelength, sigma in zip(wavelengths, sigma_to_add):
    #     trapz_samples = np.array([wavelengths[:-1], wavelengths[1:]])
    #         # sample points needed for trapezoid integration
    #     psf = gaussian(trapz_samples, central_wavelength, sigma)
    #     weights.append(psf*measure*0.5)
    # weights = np.swapaxes(weights, 0, 1)
    # apply convolution, log transform, re-sampling, and normalization
    finished_spectra = []
    for counter, spectrum in enumerate(spectra):
        # smoothed_spectrum = (weights[0, ...]*spectrum[:-1] +
        #                      weights[1, ...]*spectrum[1:]).sum(axis=1)
            # trapezoid integration of continuous convolution integral
        # interpolator = interp1d(wavelengths, smoothed_spectrum)
        interpolator = interp1d(wavelengths, spectrum)
        interpolated_spectrum = interpolator(logspaced_waves)
        edge_sigmas = target_fwhm[[0, -1]]/(2*np.sqrt(2*np.log(2)))
        edge_buffer = crop_factor*edge_sigmas
        valid = ((wavelengths.min() + edge_buffer[0] < logspaced_waves) &
                 (logspaced_waves < wavelengths.max() - edge_buffer[1]))
        interpolated_spectrum = interpolated_spectrum[valid]
        logspaced_waves = logspaced_waves[valid]
            # crop to eliminate edge effects
        norm = np.median(interpolated_spectrum)
        finished_spectra.append(interpolated_spectrum/norm)
    return np.array(finished_spectra), logspaced_waves
   

def prepare_templates_hist(spectra, wavelengths, current_fwhm, target_fwhm,
                           target_deltalog, target_range, crop_factor=5,
                           tol=10**(-12)):
    """
    Prepare a set of template spectra for use by pPXF.  Input templates are
    assumed to be sampled at identical wavelengths with identical spectral
    resolution.  Outputs spectra will be smoothed to a desired resolution,
    re-sampled on a log-spaced grid, re-binned to give the log-specific
    flux and normalized to have numerical values near 1.

    Args:
    spectra - 2D arraylike
        Array of template spectra, with spectra occupying the rows,
        units of flux/length where length matches wavelength
    wavelengths - 1D arraylike
        Wavelength sample points of spectra, units of length
    current_fwhm - float
        FWHM resolution of templates, units of matching wavelength
    target_fwhm - 1D arraylike
        Desired FWHM resolution of final template spectra, as a function
        of wavelength, units of matching wavelength
    target_deltalog - float
        Desired uniform step in log(wavelength) of final template sampling
    target_range - 1D arraylike
        Desired wavelength range of final template, as [min, max] array,
        units matching wavelengths

    Returns: finished_spectra, logspaced_waves
    finished_spectra - 2D arraylike
        Array of prepared spectra, with spectra occupying the rows,
        units of flux/log(length) where length matches wavelength
    logspaced_waves - 1D arraylike
        Wavelength sampling points of prepared spectra
    """
    # check inputs
    spectra, wavelengths, target_range, target_fwhm = (
        map(np.array, [spectra, wavelengths, target_range, target_fwhm]))
    current_fwhm, target_deltalog = (
        map(float, [current_fwhm, target_deltalog]))
    # compute new wavelength sampling
    slop = 10**(-10)  # prevent accidental extrapolation
    log_range = np.log(target_range)
    log_waves = np.arange(log_range[0]*(1.0 + slop),
                          log_range[1]*(1.0 - slop),
                          target_deltalog)
    logspaced_waves = np.exp(log_waves)
    # compute convolution weights
    fwhmsq_to_add = target_fwhm**2 - current_fwhm**2
    sigmasq_to_add = fwhmsq_to_add/(8*np.log(2))
    measure = wavelengths[1:] - wavelengths[:-1]
    weights = []
    sigma_iter = enumerate(zip(wavelengths, sigmasq_to_add))
    for n, (central_wavelength, sigmasq) in sigma_iter:
        trapz_samples = np.array([wavelengths[:-1], wavelengths[1:]])
            # sample points needed for trapezoid integration
        if sigmasq > tol:
            sigma = np.sqrt(sigmasq)
            psf = gaussian(trapz_samples, central_wavelength, sigma)
        else:
            psf = np.zeros(trapz_samples.shape)
            if n == 0:
                psf[0, 0] = 2.0/measure[0]
            elif n == (wavelengths.size - 1):
                psf[1, -1] = 2.0/measure[-1]
            else:
                psf[0, n] = 1.0/measure[n]
                psf[1, n - 1] = 1.0/measure[n - 1]
        weights.append(psf*measure*0.5)
    weights = np.swapaxes(weights, 0, 1)
    # apply convolution, log transform, re-sampling, and normalization
    finished_spectra = []
    for counter, spectrum in enumerate(spectra):
        smoothed_spectrum = (weights[0, ...]*spectrum[:-1] +
                             weights[1, ...]*spectrum[1:]).sum(axis=1)
            # trapezoid integration of continuous convolution integral
        interpolator = interp1d(wavelengths, smoothed_spectrum*wavelengths)
            # re-sample and re-bin
        interpolated_spectrum = interpolator(logspaced_waves)
        edge_sigmas = target_fwhm[[0, -1]]/(2*np.sqrt(2*np.log(2)))
        edge_buffer = crop_factor*edge_sigmas
        valid = ((wavelengths.min() + edge_buffer[0] < logspaced_waves) &
                 (logspaced_waves < wavelengths.max() - edge_buffer[1]))
        interpolated_spectrum = interpolated_spectrum[valid]
        logspaced_waves = logspaced_waves[valid]
            # crop to eliminate edge effects
        # norm = np.median(interpolated_spectrum)
        # finished_spectra.append(interpolated_spectrum/norm)
        finished_spectra.append(interpolated_spectrum)
    return np.array(finished_spectra), logspaced_waves


def in_range_union(values, ranges):
    """
    Return a mask indicating all elements of the passed wavelengths
    that lie inside one of the passed masked ranges

    Args:
    values - arraylike
        The values to be tested for inclusion in the union of the
        passed ranges
    ranges - iterable
        An iterable of ranges [lower, upper]

    Returns: in_union
    in_union - bool array
        True for all elements in values that are interior to the
        union of the intervals in ranges
    """
    in_union = np.zeros(values.shape, dtype=np.bool)
    for lower, upper in ranges:
        in_range = (lower <= values) & (values <= upper)
        in_union = in_union | in_range
    return in_union


def compute_ir(wavelengths, ir_data):
    """
    Given a file name for an instrumental resolution file containing
    a two-column wavelength, ir measurement of the resolution, along
    with the observed data wavelengths, returns the ir interpolated
    onto all of the data wavelength samples. If necessary, extrapolation
    will be done as a constant from the nearest data point.
    """
    ir = ir_data[:, 1]   # ir format: first col waves, second col fwhm A
    ir_sample_wavelenths = ir_data[:, 0]
    ir_interp_func = interp1d(ir_sample_wavelenths, ir, kind='cubic',
                              bounds_error=False, fill_value=np.nan)
    interpolated_ir = ir_interp_func(wavelengths)
    min_sample = ir_sample_wavelenths.min()
    min_sample_index = np.argmin(ir_sample_wavelenths)
    lower_extrapolation = wavelengths < min_sample
    max_sample = ir_sample_wavelenths.max()
    max_sample_index = np.argmax(ir_sample_wavelenths)
    upper_extrapolation = max_sample < wavelengths
    interpolated_ir[upper_extrapolation] = ir[max_sample_index]
    interpolated_ir[lower_extrapolation] = ir[min_sample_index]
    return interpolated_ir


def safe_int(value, fail_value=np.nan):
    try:
        return int(value)
    except:
        return fail_value


def safe_str(value, fail_value='----'):
    try:
        to_string = str(value)
        if to_string:
            return to_string
        else:
            return fail_value
    except:
        return fail_value


def safe_float(value, fail_value=np.nan):
    try:
        return float(value)
    except:
        return fail_value


def miles_spec_type_conv(value, fail_value='----'):
    string_spectype = safe_str(value, fail_value=fail_value)
    return string_spectype.replace(' ', '_')


def read_miles_index(index_filename, templates=None):
    """ Read miles library description into list """
    line_length = 74
    fields = [
        [0,   8,   "template_num",  safe_int],
        [8,   29,  "object_name",   safe_str],
        [29,  36,  "reddening",     safe_float],
        [36,  52,  "spec_type",     lambda s: safe_str(s).replace(' ', '_')],
        [52,  61,  "temperature",   safe_float],
        [61,  68,  "surf_grav",     safe_float],
        [68,  73,  "metallicity",   safe_float]]
    if templates is not None:
        templates = 1 + np.array(templates, dtype=np.int64)  # to MILES number
    star_data = []
    with open(index_filename, 'r') as index_file:
        for line_num, line in enumerate(index_file):
            comment = (line[0] == '#')
            blank_line = (line == '\n')
            data = (not comment) and (not blank_line)
            if data:
                missing = line_length - len(line)
                if missing > 0:
                    line = line + ' '*missing
                elif missing < 0:
                    raise Exception("Invalid MILES index format")
                num_left, num_right = fields[0][0:2]
                num_convert = fields[0][3]
                miles_num = num_convert(line[num_left:num_right].strip())
                if (templates is None) or (miles_num in templates):
                    current_star_data = []
                    for left, right, name, convert in fields:
                        converted = convert(line[left:right].strip())
                        current_star_data.append(converted)
                    star_data.append(current_star_data)
    return star_data


def write_results_files(params, error, chisq_dof, temp_weights, covmat,
                        temp_info, results_filename, title):
    """ Write file summarizing ppxf fit to MASSIVE spectral bin """
    with open(results_filename, 'w') as results_file:
        results_file.write("pPXF fit results: {}\n".format(title))
        results_file.write("chi^2/dof = {:7.3f}\n".format(chisq_dof))
        results_file.write("v         = {:7.3f} +- {:7.3f} km/s\n"
                           "".format(params[0], error[0]*np.sqrt(chisq_dof)))
        results_file.write("sigma     = {:7.3f} +- {:7.3f} km/s\n"
                           "".format(params[1], error[1]*np.sqrt(chisq_dof)))
        missing = 6 - params.shape[0]
        if missing > 0:  # fit less than 6 params, need to add zeros
            params = np.concatenate((params, np.zeros(missing)))
            error = np.concatenate((error, np.zeros(missing)))
        elif missing < 0:
            raise Exception("Invalid bestfit parameter format")
        for h_num, (h_value, h_error) in enumerate(zip(params[2:], error[2:])):
            results_file.write("h{}        = {:7.3f} +- {:7.3f}\n"
                               "".format(3 + h_num, h_value,
                                         h_error*np.sqrt(chisq_dof)))
        results_file.write("covariance matrix:\n")
        for row in covmat:
            for element in row:
                results_file.write("{:6.3e}  ".format(element))
            results_file.write("\n")
        results_file.write("templates: {}/{} nonzero\n"
                          "".format(np.sum(temp_weights > 0),
                                    temp_weights.shape[0]))
        results_file.write("weight%\tmiles#\tname\t\ttype"
                          "\t\t\t\tTeff\tlogg\t[Fe/H]\n")
        sorted_template_num = np.argsort(temp_weights)[::-1]
        percentage_weights = temp_weights*100.0/np.sum(temp_weights)
        for template_num in sorted_template_num:
            if temp_weights[template_num] > 0:
                results_file.write("{:5.2f}\t{}\n"
                                   "".format(percentage_weights[template_num],
                                             temp_info[template_num]))
    return


def plot_results(spectrum, model, noise, wavelengths, masked,
                 add_poly, mult_poly, chisq_dof, title, filename):
    """
    Plot ppxf fit results with a two-panel model/data and residuals plot
    """
    laptop_height = 8.1
    figsize = (2.075*laptop_height, laptop_height)
    fig = plt.figure(figsize=figsize)
    subplot_grid = gridspec.GridSpec(2, 1, height_ratios=[4,1])
    ax0 = plt.subplot(subplot_grid[0])
    ax1 = plt.subplot(subplot_grid[1], sharex=ax0)
    upper = spectrum + noise
    lower = spectrum - noise
    ax0.fill_between(wavelengths, lower, upper,
                       color='k', alpha=0.7)
    ax0.plot([], [], color='k', label='Data', linewidth=1.5)
    ax0.plot(wavelengths, model, color='r', linewidth=1.5,
               label='pPXF Fit')
    if np.size(add_poly) > 0:
        continuum = np.polynomial.legendre.legval(
            np.linspace(-1, 1, wavelengths.shape[0]), add_poly)
        ax0.plot(wavelengths, continuum, 'g', linewidth=1.5,
                 label="Additive Correction")
    if np.size(mult_poly) > 0:
        mult_continuum = np.polynomial.legendre.legval(
            np.linspace(-1, 1, wavelengths.shape[0]),
            np.concatenate(([1], mult_poly)))
        shifted_mult_continuum = mult_continuum/np.mean(mult_continuum)
        ax0.plot(wavelengths, mult_continuum, 'm', linewidth=1.5,
                 label="Multiplicative Correction (mean = 1)")
    valid_max = np.max([model[~masked], upper[~masked]])
    valid_min = np.min([model[~masked], lower[~masked]])
    margin = (valid_max - valid_min)*0.05/(1 - 0.05)
        # set margin to be 5% of the total plot height
    ax0.set_ylim([valid_min - margin, valid_max + margin])
    ax0.set_title("pPXF Fit to {}".format(title))
    ax0.set_ylabel("Flux [arbitrary]")
    ax0.plot([],[], color='b', label='Excluded', linewidth=1.5)
    leg = ax0.legend(loc='best', prop={'size':12},
                     title="chisq/dof = {:.4f}".format(chisq_dof))
    leg_title = leg.get_title()
    leg_title.set_fontsize(12)
    residuals = (spectrum - model)/(spectrum+1e-19)
    ax1.axhline(0.0, color='k')
    ax1.plot(wavelengths, residuals, 'g')
    number_masked = np.sum(masked)
    masked_pixels = np.arange(masked.shape[0])[masked]
    good_pixels = np.arange(masked.shape[0])[~masked]
    if number_masked > 0:
        last_masked_pixel = masked_pixels[0]
        masked_regions = [[last_masked_pixel]]
        for pixel in masked_pixels[1:]:
            if pixel - last_masked_pixel == 1:
                masked_regions[-1].append(pixel)
            else:
                masked_regions.append([pixel])
            last_masked_pixel = pixel
        for region in masked_regions:
            ax1.plot(wavelengths[region], residuals[region], 'b')
            ax1.axvline(x=wavelengths[region[0]], color='b')
            ax1.axvline(x=wavelengths[region[-1]], color='b')
            ax0.axvline(x=wavelengths[region[0]], color='b')
            ax0.axvline(x=wavelengths[region[-1]], color='b')
    axis_scale = 1.1
    resid_limit = np.absolute(residuals[good_pixels]).max()
    ax1.set_ylim([-resid_limit*axis_scale, resid_limit*axis_scale])
    ax1.set_title("Fractional Residuals")
    ax1.set_xlabel("Wavelength (A)")
    ax1.set_xlim([wavelengths[~masked].min(), wavelengths[~masked].max()])
    fig.savefig(filename, dpi=100)
    plt.close(fig)
    return

def compute_projected_confidences(prob_draws, fraction=0.683):
    """
    Given a set of draws from an assumed Gaussian distribution in D
    dimensions, this computes the D-ellipsoid containing the given
    probability percentile, and then projects that ellipsoid onto the
    planes given by each individual parameter.

    Args:
    prob_draws - 2D arraylike
        A set of N draws from a D-dimension distribution, with each
        draw occupying a row, i.e. the shape of prob_draws is (N, D)
    fraction - float, default=0.683
        The fraction of probability weight to be enclosed by the
        D-ellipse.  Default is 0.683, but note that this does not quite
        give a '1-sigma' ellipsoid: the weight enclosed by the covariance
        ellipsoid of a Gaussian distribution depends on dimension and is
        decreasing.  In 1D, 1 sigma corresponds to 68.3% cconfidence, but
        in higher dimension 1 sigma encloses less than 68.3% of the
        probability weight.  This code works off of percentiles rather
        than sigma, so the ellipsoid returned is in general going to be
        some larger multiple of the 1 sigma ellipse than as naively
        expected from the 1D case.

    Returns: metric, samples, all_hulls
    covariance - 2D arraylike
        The covariance matrix of the Gaussian describing the samples.
    intervals - 1D arraylike
        The half-width of the projected confidence intervals
    """
    # get 6D confidence ellipse
    covariance = np.cov(prob_draws.T)  # normalized
    metric = np.linalg.inv(covariance)  # Mahalanobis metric
    center = np.median(prob_draws, axis=0)
    mdist_sq = []
    for num, row in enumerate(prob_draws):
        shifted = row - center
        mdist_sq.append(np.dot(shifted.T, np.dot(metric, shifted)))
    conf_mdist_sq = np.percentile(mdist_sq, fraction*100)
    max_displacments = np.sqrt(np.diag(covariance*conf_mdist_sq))
    return covariance, max_displacments


#################################################

if __name__ == '__main__':
    target_info_filename = sys.argv[1]
    fitting_info_filename = sys.argv[2]
    # read setup data
    target_info = read_target_info(target_info_filename)
    fitting_info = read_fitting_info(fitting_info_filename)
    
    
    master_fits = "{}/{}".format(target_info["data_dir"], target_info["fiber_data"])
    ir_file = "{}/{}".format(target_info["data_dir"], target_info["ir_file"])


    center = np.array([target_info['center_ra'],target_info['center_dec']])
    pa = target_info['pa']; ma_ang = 90.0 - pa; ma = ma_ang*np.pi/180

    master = fits.open(master_fits)
    coords = master[3].data
    spectra = master[0].data
    noise = master[1].data
    waves = master[2].data; logscales = np.log(waves[:,1]/waves[:,0])

    masked_fibers = [a for a in target_info["rm_fiber"] if a != -1]  #fiber indices start at 0
    spectra[masked_fibers,...] = 1e19
    noise[masked_fibers,...] = 1e19
    


    #masking
    mask = ((np.absolute(spectra) > 10**4) | (spectra <= 0))
    #mask = ((spectra < 0) | (spectra > 10**3))

    # parse data files
    coords, fluxes, flux_noise = parse_fiber_data("{}/{}".format(target_info["data_dir"], target_info["fiber_data"]),center)

    # binning
    fibers_in_bin, binned_data, radial_sets = make_bins.determine_polar_bins_folded(spectra,noise,mask,coords, aspect_ratio = 1.5, s2n_limit = 20, major_axis = ma)

    
    outputfile = open('s2n-'+target_info['name']+'.dat','w+')
    for binned_datum in binned_data:
        spectrumb = binned_datum[0,...]
        noiseb = binned_datum[1,...]
        maskb = binned_datum[2,...]

        spectrumb = np.ma.array(spectrumb,mask=maskb)
        noiseb = np.ma.array(noiseb,mask=maskb)

        s2n = np.ma.mean(spectrumb/noiseb)
        outputfile.write(str(s2n) + '\n')
    outputfile.close()

    

    nbinned_data = []

    for ii in range(len(binned_data)):
        datum = list(binned_data[ii])
        datum.insert(2,waves[0,:])
        nbinned_data = nbinned_data + [datum]

    print len(fibers_in_bin)


    # make full galaxy bin
    full_spectrum, full_noise, full_mask = make_bins.combine_spectra_ivar(spectra,noise,mask)
    #from pylab import*
    #plot(waves[0,~full_mask],full_spectrum[~full_mask])
    #plot(waves[0,:],full_noise)
    #show()
    full_data = [full_spectrum, full_noise, waves[0,:], full_mask]


    # bin computations
    weighted_positions, bin_fluxes = compute_bindata_folded(fibers_in_bin, coords, fluxes, ma_ang)
    all_bin_data = [full_data] + nbinned_data
    all_bin_data = np.array(all_bin_data)
    #sys.exit()

    outputfile = open(target_info["name"]+'-bincoords.dat','w+')
    for ii in range(np.shape(weighted_positions)[0]):
        outputfile.write(str(ii+1) + '\t' + str(weighted_positions[ii,0]) + '\t' + str(weighted_positions[ii,1]) + '\n')
    outputfile.close()
    
    #sys.exit()
# plot fibers and bins

    bin_outlines = []

    colors = ['b', 'g', 'r', 'c', 'm']
    used_fibers = []
    fiber_radius = 2.08
    fig, ax = plt.subplots()
    for n, fibers  in enumerate(fibers_in_bin):
        bin_color = colors[n % len(colors)]
        for fiber in fibers:
            used_fibers.append(fiber)
            ax.add_patch(Circle(coords[fiber, :], fiber_radius,
                            facecolor=bin_color, zorder=0,
                            linewidth=0.25, alpha=0.8))
        ax.set_aspect('equal')
    for unused_fiber in range(coords.shape[0]):
        if unused_fiber not in used_fibers:
            ax.add_patch(Circle(coords[unused_fiber, :], fiber_radius,
                            facecolor='k', zorder=0,
                            linewidth=0.25, alpha=0.3))
    for fibers in fibers_in_bin:
        if len(fibers) > 1:
            break
        if len(fibers) == 1:
            bin_outlines.append(None)

    for n, ((rmin, rmax), angle_bounds) in enumerate(radial_sets):
        for m, (amin, amax) in enumerate(angle_bounds):
            num = n + m
            bin_poly = polar_box(rmin, rmax, amin*180.0/np.pi, amax*180.0/np.pi)
            bin_outlines.append(bin_poly)
            ax.add_patch(PolygonPatch(bin_poly, facecolor='none',
                                  linestyle='solid', linewidth=1.5))
    plt.plot(weighted_positions[:,0],weighted_positions[:,1],'ko')

    ax.set_aspect("equal")
    ax.set_title("{} virus-p fibers".format(507), fontsize=16)
    ax.set_ylabel("arcsec")
    ax.set_xlabel("arcsec")
    ax.autoscale_view()
    plt.savefig("{}-fibermap.png".format(507))
    plt.show()

    #sys.exit()
        # get ir files
    print "Getting IR files."
    binned_ir = []
    with open(ir_file, 'rb') as pickled:
        pickled_ir = pickle.load(pickled)
    total_flux = np.sum(fluxes)
    binned_waves = np.sum((pickled_ir[0, ...].T)*fluxes, axis=1)/total_flux
    binned_fwhm = np.sum((pickled_ir[1, ...].T)*fluxes, axis=1)/total_flux
    fiber_ir = np.array([binned_waves, binned_fwhm]).T
    binned_ir.append(fiber_ir)
    for bin_num, fibers in enumerate(fibers_in_bin):
        fiber_flux = fluxes[fibers]
        total_flux = np.sum(fiber_flux)
        waves = pickled_ir[0, fibers, :]
        fwhm = pickled_ir[1, fibers, :]
        binned_waves = np.sum((waves.T)*fiber_flux, axis=1)/total_flux
        binned_fwhm = np.sum((fwhm.T)*fiber_flux, axis=1)/total_flux
        fiber_ir = np.array([binned_waves, binned_fwhm]).T
        np.savetxt("ir-bin{:02d}".format(bin_num), fiber_ir,
                   fmt=["%8.6f", "%6.4f"], delimiter=" "*4)
        binned_ir.append(fiber_ir)
    binned_ir = np.array(binned_ir)


    # pickle bin data - use old format - HACK
    pickled_bins_filename = "{}-bindata.p".format(target_info["name"])
    old_bins = list(zip(fibers_in_bin, bin_outlines))
    data_to_save = [old_bins, weighted_positions, bin_fluxes, fluxes, coords]
    with open(pickled_bins_filename, 'wb') as pickled:
        pickle.dump(data_to_save, pickled)

    # gather template library
    print "Preparing Templates."
    templates = [f for f in os.listdir(fitting_info['template_dir'])
                 if f != fitting_info["template_index"]]
    template_spectra = []
    templates_used = []
    for temp_num, template in enumerate(templates):
        template_path = "{}/{}".format(fitting_info['template_dir'], template)
        template_data = np.loadtxt(template_path)
        # this assume the MILES format of col 0 waves, col 1 spectrum
        template_spectra.append(template_data[:, 1])
        if temp_num == 0:  # only need this once - assume all else identical
            template_wavelengths = template_data[:, 0]
        # get template number
        template_number = int(template[1:-1])  # MILES format: m####V
        templates_used.append(template_number)
    template_spectra = np.array(template_spectra)
    template_range = [template_wavelengths.min(), template_wavelengths.max()]
    temp_index_filename = "{}/{}".format(fitting_info["template_dir"],
                                         fitting_info["template_index"])
    full_template_info = read_miles_index(temp_index_filename)
    template_info = [t for n, t in enumerate(full_template_info)
                       if (n + 1) in templates_used]
    # fit full galaxy for template library reduction
    full_spectrum = all_bin_data[0, 0, :]
    full_noise = all_bin_data[0, 1, :]
    full_wavelengths = all_bin_data[0, 2, :]
    full_badpixels = all_bin_data[0, 3, :].astype(bool)
    full_logscale = logscales[0]
    full_ir = binned_ir[0, ...]
        # masking
    masked_pixels = in_range_union(full_wavelengths, fitting_info["mask_observed"])
    masked_pixels = masked_pixels | full_badpixels
    good_pixels = np.arange(full_wavelengths.shape[0])[~masked_pixels]
        # prepare templates
    interpolated_ir = compute_ir(template_wavelengths, full_ir)
    prepared_templates, prepared_template_waves = (prepare_templates(template_spectra, template_wavelengths,fitting_info["template_fwhm"], interpolated_ir,full_logscale, template_range))

        # run fit
    print "now running fit"
    log_temp_initial = np.log(prepared_template_waves.min())
    log_galaxy_initial = np.log(full_wavelengths.min())
    velocity_offset = (log_temp_initial - log_galaxy_initial)*speed_of_light
    full_velscale = full_logscale*speed_of_light
    moments_to_fit = fitting_info["moments_to_fit"];
    if isinstance(moments_to_fit,int):
        moments_to_fit = [moments_to_fit]
    poly_degrees = fitting_info["additive_degree"]
    bias = fitting_info["bias"]
    mult_degree = fitting_info["multiplicative_degree"]
    nonzero_weight = np.zeros(prepared_templates.shape[0], dtype=bool)
    full_galaxy_gaussians = []
    for num_moments in moments_to_fit:
        guess = ([fitting_info["v_guess"],
                  fitting_info["sigma_guess"]] +
                 [fitting_info["hn_guess"]]*(num_moments - 2))
        ppxf_fitter = ppxf(prepared_templates.T,  # template spectra in columns
                           full_spectrum, full_noise, full_velscale,
                           guess, goodpixels=good_pixels,
                           moments=num_moments, degree=poly_degrees,
                           vsyst=velocity_offset, bias=bias,
                           mdegree=mult_degree, plot=False, quiet=True)
        print "fit full galaxy"
        bestfit_params = ppxf_fitter.sol
        full_galaxy_gaussians.append(bestfit_params[:2])
        bestfit_paramerror = ppxf_fitter.error
        bestfit_model = ppxf_fitter.bestfit
        chisq_dof = ppxf_fitter.chi2
        bestfit_tempweights = ppxf_fitter.weights
        bestfit_fixedtemplate = ppxf_fitter.fixedtemplate
        nonzero_weight = nonzero_weight | (bestfit_tempweights > 0.0)
            # output results
        num_params = bestfit_params.shape[0]
        corr = np.ones((num_params, num_params))*np.nan
            # python ppxf doesn't output a correlation matrix, need to mc it
        fit_title = "bin00_{}-p{}-fullmiles".format(target_info['name'],
                                                    num_moments)
        write_results_files(bestfit_params, bestfit_paramerror, chisq_dof,
                            bestfit_tempweights, corr, template_info,
                            "{}-results.dat".format(fit_title), fit_title)
        plot_results(full_spectrum, bestfit_model, full_noise, full_wavelengths,
                     masked_pixels, bestfit_polyweights, chisq_dof,
                     fit_title, "{}.png".format(fit_title))
    # fit all bins with reduced library
    reduced_template_spectra = template_spectra[nonzero_weight, :]
    reduced_template_numbers = np.arange(nonzero_weight.shape[0])[nonzero_weight]
    reduced_template_info = [info for info, nonzero in
                             zip(template_info, nonzero_weight) if nonzero]
    bin_iteration = enumerate(zip(all_bin_data, logscales))
    for bin_number, (bin_data, logscale) in bin_iteration:
        print "fit bin number " + str(bin_number)
        target_range = template_range  # hack, need to add proper cropping via sigma
        bin_spectrum = bin_data[0, :]
        bin_noise = bin_data[1, :]
        bin_wavelengths = bin_data[2, :]
        bin_badpixels = bin_data[3, :].astype(bool)
        bin_ir = binned_ir[bin_number, ...]
        
        for ii in range(len(bin_noise)):
            if bin_noise[ii] <= 0:
                bin_noise[ii] = 1e19 #make the noise vector positive, albeit absurdly so

        # save binned spectra
        binned_spectra_filename = ("bin{:02d}_{}_python.fits"
                                   "".format(bin_number, target_info["name"]))
        fits.writeto(binned_spectra_filename, bin_data, clobber=True)
        # masking
        masked_pixels = in_range_union(bin_wavelengths,
                                       fitting_info["mask_observed"])
        masked_pixels = masked_pixels | bin_badpixels
        good_pixels = np.arange(bin_wavelengths.shape[0])[~masked_pixels]
        # template prep
        interpolated_ir = compute_ir(template_wavelengths, bin_ir)
        prepared_templates, prepared_template_waves = (
            prepare_templates(reduced_template_spectra, template_wavelengths,
                              fitting_info["template_fwhm"], interpolated_ir,
                              logscale, target_range))
        # actual fit
        log_temp_initial = np.log(prepared_template_waves.min())
        log_galaxy_initial = np.log(bin_wavelengths.min())
        velocity_offset = (log_temp_initial - log_galaxy_initial)*speed_of_light
        velscale = logscale*speed_of_light
        for num_moments, gaussian_guess in zip(moments_to_fit,
                                               full_galaxy_gaussians):
            hn_guesses = np.ones(num_moments - 2)*fitting_info["hn_guess"]
            guess = np.concatenate((gaussian_guess, hn_guesses))
            ppxf_fitter = ppxf(prepared_templates.T,  # template spectra in columns
                               bin_spectrum, bin_noise, velscale,
                               guess, goodpixels=good_pixels,
                               moments=num_moments, degree=poly_degrees,
                               vsyst=velocity_offset, bias=bias,
                               mdegree=mult_degree, plot=False, quiet=True)
            bestfit_params = ppxf_fitter.sol
            bestfit_paramerror = ppxf_fitter.error
            bestfit_model = ppxf_fitter.bestfit
            chisq_dof = ppxf_fitter.chi2
            bestfit_tempweights = ppxf_fitter.weights
            bestfit_polyweights = ppxf_fitter.polyweights
            # error simulation
            num_sims = 100
            samples = np.zeros((num_sims, bestfit_params.shape[0]))
            for trial in xrange(num_sims):
                noise_draw = np.random.randn(*bin_spectrum.shape)  # uniform guassian
                simulated_galaxy = bin_spectrum + noise_draw*bin_noise
                ppxf_fitter = ppxf(prepared_templates.T,  # template spectra in columns
                                   simulated_galaxy, bin_noise, velscale,
                                   guess, goodpixels=good_pixels,
                                   moments=num_moments, degree=poly_degrees,
                                   vsyst=velocity_offset, bias=0.0,  # bias 0 for noise
                                   mdegree=mult_degree, plot=False, quiet=True)
                samples[trial, :] = ppxf_fitter.sol

            outputfilename = "bin{:02d}_{}-p{}".format(bin_number,
                                                       target_info['name'],
                                                       fitting_info["moments_to_fit"]) + '-losvdsamples.dat'
            outputfile = open(outputfilename,'w+')

            for sample in samples:
                for numero in sample:
                    outputfile.write('{:7.3f}'.format(numero) + '\t')
                outputfile.write('\n')
            outputfile.close()

            covmat, sim_paramerror = compute_projected_confidences(samples)

            ###since there is no covariance matrix

#            num_params = bestfit_params.shape[0]
#            covmat = np.ones((num_params, num_params))*np.nan
            ###delete when done!


            fit_title = "bin{:02d}_{}-p{}".format(bin_number,
                                                  target_info['name'],
                                                  num_moments)
            write_results_files(bestfit_params, sim_paramerror, chisq_dof,
                                bestfit_tempweights, covmat, reduced_template_info,
                                "{}-results.dat".format(fit_title), fit_title)
            plot_results(bin_spectrum, bestfit_model, bin_noise, bin_wavelengths,
                         masked_pixels, bestfit_polyweights, chisq_dof,
                         fit_title, "{}.png".format(fit_title))
#change bestfit_paramerror to sim_paramerror



