""" Module for computing and manipulation spectral resolutions """


import numpy as np

import utilities as utl
import fitting.series as sf
import massivepy.gausshermite as gh
import massivepy.constants as const


def fit_arcset(wavelengths, arcs, line_centers, fwhm_guess, fit_scale=10):
    """
    Fit a set of arc spectra, extracting the centers and fwhm of each
    line. The lines are fit with a Gaussian profile.

    Args:
    wavelengths - 1d arraylike
        The wavelength sampling of the arc spectra.
    arcs - 2d arraylike
        The arc spectra, shaped (N, M), where M is the length of
        the above passed wavelength sampling. Each row is a spectrum.
    line_centers - 1d arraylike
        The central values of wavelength of the lines to fit, units
        must match that of passed wavelengths.
    fwhm_guess - 1d arraylike
        The initial estimate fwhm of each line, units
        must match that of passed wavelengths.
    fit_scale - float, default = 10
        This scale sets the size of the fitting region used for each
        line. The region, centered on the line, will be greater than
        fit_scale times the best-fit Gaussian sigma of the line. If
        a line's region encompassed another line, then the sum of both
        lines will be fit in the union of their fit regions.

    Returns: spec_resolution
    spec_resolution - array of size (N, P, 2)
        Measured line widths, with N the number of passed spectra and
        P the number of lines fit. spec_resolution[i, j, 0] give
        the central wavelength of the jth line in the ith spectrum,
        and spec_resolution[i, j, 1] gives its FWHM. The units of
        both match the units of the passed wavelength samples.
    """
    arcs = np.asarray(arcs, dtype=float)
    arcs = (arcs.T/np.max(arcs, axis=1)).T
        # norm each arc spectra to have max value = 1 - simplifies fitting
    waves = np.asarray(wavelengths, dtype=float)
    num_arcs, num_samples = arcs.shape
    if waves.shape != (num_samples,):
        raise ValueError("Invalid wavelength shape {}, must match "
                         "arc sampling with shape ({},)"
                         "".format(waves.shape, num_samples))
    line_centers = np.asarray(line_centers, dtype=float)
    num_lines, = line_centers.shape
    # define Gaussian line models
    gauss_model = lambda x, p: p[0]*gh.unnormalized_gausshermite_pdf(x, p[1:])
    gauss_center = lambda p: p[1]
    gauss_fitwidth = lambda p: float(fit_scale)*p[2]
    sigma_guess = float(fwhm_guess)/const.gaussian_fwhm_over_sigma
    initial_sigmas = np.ones(num_lines)*sigma_guess
    initial_heights = np.ones(num_lines)
    all_intital_params = zip(initial_heights, line_centers, initial_sigmas)
    # fit lines
    spec_resolution = np.zeros((num_arcs, num_lines, 2))
    for arc_iter, arc in enumerate(arcs):
        fitter = sf.SeriesFit(waves, arc, gauss_model, gauss_fitwidth,
                              gauss_center, all_intital_params,
                              sf.leastsq_lma) # lma least-squares fit
        fitter.run_fit()
        bestfit_centers, bestfit_sigmas = fitter.current_params[:, 1:].T
            # param order: line height, center, sigma
        bestfit_fwhm = bestfit_sigmas*const.gaussian_fwhm_over_sigma
        spec_resolution[arc_iter, :, 0] = bestfit_centers
        spec_resolution[arc_iter, :, 1] = bestfit_fwhm
    return spec_resolution