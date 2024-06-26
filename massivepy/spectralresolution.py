""" Module for computing and manipulation spectral resolutions """


import numpy as np
import astropy.io.fits as fits

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
    dt = {'names':['center','fitcenter','fwhm','height'],
          'formats':4*[np.float64]}
    spec_resolution = np.zeros((num_arcs, num_lines),dtype=dt)
    for arc_iter, arc in enumerate(arcs):
        fitter = sf.SeriesFit(waves, arc, gauss_model, gauss_fitwidth,
                              gauss_center, all_intital_params,
                              sf.leastsq_lma) # lma least-squares fit
        fitter.run_fit()
        bestfit_heights,bestfit_centers,bestfit_sigmas=fitter.current_params.T
        bestfit_fwhm = bestfit_sigmas*const.gaussian_fwhm_over_sigma
        spec_resolution['center'][arc_iter, :] = line_centers
        spec_resolution['fitcenter'][arc_iter, :] = bestfit_centers
        spec_resolution['fwhm'][arc_iter, :] = bestfit_fwhm
        spec_resolution['height'][arc_iter, :] = bestfit_heights
    return spec_resolution

def specres_for_galaxy(inst_waves, fwhm, gal_waves, redshift):
    """
    Take the results from fit_arcset and put them in galaxy terms.
    1) Shift from instrument frame to galaxy rest frame.
    2) Interpolate to galaxy wavelength samples.
    Assumes a set of spec_resolution results (== set of fibers), but with
     uniform redshift and wavelength sampling across all fibers.
    Assumes spec_resolution is an array containing at least two columns,
     'centers' and 'fwhm' - other columns are ignored.
    Returns a set of ir arrays (one for each fiber).
    """
    nfibers, nlines = fwhm.shape
    npixels = len(gal_waves)
    ir_set = np.zeros((nfibers,npixels))
    for ifiber in range(nfibers):
        gal_wavesamples = inst_waves[ifiber,:]/(1+redshift)
        fwhm_samples = fwhm[ifiber,:]/(1+redshift)
        interpolator = utl.interp1d_constextrap(gal_wavesamples,fwhm_samples)
        ir_set[ifiber] = interpolator(gal_waves)
    return ir_set

def save_specres(path,samples,source_metadata):
    nfibers, nlines = samples.shape
    header = ('Columns are as follows:'
              '\n fiberiter, (for each line) center, fitcenter, fwhm, height'
              '\nMetadata is as follows:'
              '\n {headermeta}'
              '\nAdditional comments...'
              '\n    This file contains fits of arc frames for each fiber'
              '\n    interpolated from {nlines} arc lamp lines'
              '\n    reported in instrument rest frame'
              ''.format)
    for key in source_metadata:
        if len(key) > 20:
            raise Exception('Overly long metadata key: {}'.format(key))
    metalist = ['{spacing}{k}: {v}'.format(spacing=(21-len(k))*' ',k=k,v=v)
                for k,v in sorted(source_metadata.iteritems())]
    header = header(headermeta='\n '.join(metalist),nlines=nlines)
    #header += "\nSource file: {}".format(source_metadata['rawfile'])
    #header += "\n from {}".format(source_metadata['rawdate'])
    savearray = np.zeros((nfibers, 1+4*nlines))
    fmt = ['%1i'] + nlines*['%-7.6g','%-7.6g','%-7.4g','%-7.4g']
    savearray[:,0] = range(nfibers)
    savearray[:,1::4] = samples['center']
    savearray[:,2::4] = samples['fitcenter']
    savearray[:,3::4] = samples['fwhm']
    savearray[:,4::4] = samples['height']
    np.savetxt(path, savearray, fmt=fmt, delimiter='\t', header=header)
    return

def read_specres(path):
    textarray = np.genfromtxt(path)
    nfibers, ncols = textarray.shape
    nlines = (ncols-1)/4
    dt = {'names':['fiberiter','center','fitcenter','fwhm','height'],
          'formats':[int] + 4*[np.float64]}
    ir = np.zeros((nfibers,nlines),dtype=dt)
    ir['fiberiter'] = textarray[:,0][:,np.newaxis]
    ir['center'] = textarray[:,1::4]
    ir['fitcenter'] = textarray[:,2::4]
    ir['fwhm'] = textarray[:,3::4]
    ir['height'] = textarray[:,4::4]
    return ir
