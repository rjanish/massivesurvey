"""
Convert 2-column .dat spectra into .fits compatible with ppxf driver
"""


import os
import sys
import datetime as dt

import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import astropy.units as units
import astropy.constants as constants

import utilities as utl


target_name = sys.argv[1]
data_dir = sys.argv[2]
data_re = sys.argv[3]
sys_velocity = float(sys.argv[4])  # km/s of galaxy
valid_range = [float(sys.argv[5]), float(sys.argv[6])]  # rest frame A
ir_suffix = sys.argv[7]
c_kms = constants.c.to(units.km/units.s).value  # km/s

data_paths = utl.re_filesearch(data_re, data_dir)[0]  # discard re match objects
print "converting {} files in {}:".format(len(data_paths), data_dir)
for path in data_paths:
    print " {}".format(path)
    data = np.loadtxt(path)
    # assumed format: array of shape (num_pix, 3), col 0 is wavelengths
    # col 1 is spectra, and col 2 is binary good pixels indicator (1 = good)
    waves, spectrum = data[:, :2].T
    good_pixels = data[:, 2].astype(bool)
    more_goodpix = np.isfinite(spectrum) & (spectrum > 0)
        # apply more strict pixel mask - Nicholas's only indicates sky lines
    good_pixels = more_goodpix & good_pixels
    bad_pixels = ~good_pixels
    poisson_noise_approx = np.sqrt(spectrum[good_pixels])
    noise = np.median(poisson_noise_approx)*np.ones(spectrum.shape)
        # assume constant noise spectrum, with ~poisson magnitude

    # log rebin and resample
    slop = 10**(-10)  # prevent rounding-error extrapolation
    delta_wave = waves[1] - waves[0]
    good_waves = waves[good_pixels]
    start, stop = good_waves.min() + slop, good_waves.max() - slop
        # do not re-sample over bad pixels at the ends of the region
    num_samples = np.sum((start <= waves) & (waves <= stop))
        # does not alter resolution
    logwaves = np.exp(np.linspace(np.log(start), np.log(stop), num_samples))
    good_spectrum = spectrum[good_pixels]  # only interp between valid data
    spec_log_func = interp.interp1d(good_waves, good_spectrum*good_waves)
    spec_log = spec_log_func(logwaves)
    good_noise = noise[good_pixels]
    noise_log_func = interp.interp1d(good_waves, good_noise*good_waves)
    noise_log = noise_log_func(logwaves)
    # Algorithm Note:
    # The old bad_pixel array no longer lines up with the new re-sampled
    # spectrum, as we've mixed up the wavelength -> pixel mapping.
    # I'll assume that each old bad pixel corresponds to a bad wavelength
    # region of +- old_pixel_spacing, and I'll mask all new values
    # that occur within such a region
    bad_wave_centers = waves[bad_pixels]
    log_bad_pixels = np.zeros(logwaves.shape, dtype=bool)
    for wave_index, wave in enumerate(logwaves):
        dist_to_bad_pixel = np.absolute(wave - bad_wave_centers)
        if np.any(dist_to_bad_pixel < delta_wave):
            log_bad_pixels[wave_index] = True

    # # check masking transfer
    # rel_norm = np.max(spec_log)/np.max(spectrum)
    # plt.plot(waves, spectrum*rel_norm, alpha=0.4, linestyle='-', marker='.',
    #          color='k', label='original')
    # original_bad = np.where(~good_pixels)[0]
    # plt.plot(waves[original_bad], spectrum[original_bad]*rel_norm, alpha=0.9,
    #          linestyle='', marker='o', color='k', label='original masked')
    # plt.plot(logwaves, spec_log, alpha=0.4, linestyle='-', marker='.',
    #          color='r', label='log')
    # log_bad = np.where(log_bad_pixels)[0]
    # plt.plot(logwaves[log_bad], spec_log[log_bad], alpha=0.9,
    #          linestyle='', marker='o', color='r', label='log masked')
    # plt.legend(loc='best')
    # plt.show()
    # sys.exit()

    # shift to galaxy rest frame
    doppler_factor = 1.0 + sys_velocity/c_kms
    logwaves = logwaves/doppler_factor
    # crop and normalize
    valid = (valid_range[0] < logwaves) & (logwaves < valid_range[1])
    log_bad_pixels = log_bad_pixels[valid]
    spec_log = spec_log[valid]
    noise_log = noise_log[valid]
    logwaves = logwaves[valid]
    norm = np.median(spec_log[~log_bad_pixels])
    spec_log /= norm
    noise_log /= norm

    # # check shifting
    # rel_norm = np.max(spec_log)/np.max(spectrum)
    # plt.plot(waves, spectrum*rel_norm, alpha=0.4, linestyle='-', marker='.',
    #          color='k', label='original')
    # original_bad = np.where(~good_pixels)[0]
    # plt.plot(waves[original_bad], spectrum[original_bad]*rel_norm, alpha=0.9,
    #          linestyle='', marker='o', color='k', label='original masked')
    # plt.plot(logwaves, spec_log, alpha=0.4, linestyle='-', marker='.',
    #          color='r', label='log')
    # log_bad = np.where(log_bad_pixels)[0]
    # plt.plot(logwaves[log_bad], spec_log[log_bad], alpha=0.9,
    #          linestyle='', marker='o', color='r', label='log masked')
    # plt.legend(loc='best')
    # plt.show()
    # sys.exit()

    binned_data = np.array([spec_log, noise_log, logwaves, log_bad_pixels])
    hdu = fits.PrimaryHDU(binned_data)  # uses a minimal default header
    hdu.header["COMMENT"] = "GMOS data reduced by Nicholas McConnell"
    hdu.header["COMMENT"] = "Target: {}".format(target_name)
    hdu.header["COMMENT"] = "Columns:"
    hdu.header["COMMENT"] = "0 - spectrum: flux/log(wavelength) [arbitrary]"
    hdu.header["COMMENT"] = ("Normalized to have a numerical value of the "
                             "median over pixels of 1")
    hdu.header["COMMENT"] = ("1 - noise: same units Column 0, "
                             "flat Poisson estimate")
    hdu.header["COMMENT"] = "2 - wavelengths: log-spaced [A]"
    hdu.header["COMMENT"] = "3 - valid pixel indicator: 1 = good, 0 = bad"
    hdu.header["COMMENT"] = ("Spectrum, noise, and wavelengths "
                             "use galaxy rest-frame wavelengths")
    hdu.header["COMMENT"] = ("Rest-frame values computed by shifting "
                             "observations by {} km/s".format(sys_velocity))
    hdu.header["COMMENT"] = ("Converted to current format by Ryan Janish, "
                             "{}".format(dt.datetime.date(dt.datetime.now())))
    bin_name = os.path.splitext(os.path.basename(path))[0] # drop dirs and extension
    new_spectrum_filename = "{}_{}.fits".format(bin_name, target_name)
    new_spectrum_path = os.path.join(data_dir, new_spectrum_filename)
    hdu.writeto(new_spectrum_path, clobber=True)
    # write assumed-constant ir
    ir_path = os.path.join(data_dir, "{}_{}".format(bin_name, ir_suffix))
    print " {}".format(ir_path)
    ir_samples = utl.fits_quickread(ir_path)[0][0][:, :2]
    first_sample = ir_samples[:, 0].min()
    last_sample = ir_samples[:, 0].max()
    slop = 10**(-10)
    low = waves < first_sample*(1 + slop)
    high = waves > last_sample*(1 - slop)
    middle = ~low & ~high
    interped_ir = np.zeros(waves.shape)
    interped_ir[low] = ir_samples[0, 1]
    interped_ir[high] = ir_samples[-1, 1]
    ir_func = interp.interp1d(ir_samples[:, 0], ir_samples[:, 1])
    interped_ir[middle] = ir_func(waves[middle])
    if np.sum(interped_ir <= 1.0) > 0:
        raise Exception
    ir = np.array([waves, interped_ir]).T
    ir_header = ("{} {} instrumental resolution\n"
                 "fwhm interpolated by Ryan Janish from "
                 "fits to OH sky lines by Nicholas McConnell\n"
                 "col 1: wavelength, A\n"
                 "col 2: gaussian fwhm, A".format(target_name, bin_name))
    ir_filename = "{}_{}_ir.txt".format(bin_name, target_name)
    new_ir_path = os.path.join(data_dir, ir_filename)
    np.savetxt(new_ir_path, ir, delimiter='  ', header=ir_header)
    print 'saved'