"""
Test basic functionality of IFUspectrum
"""


import functools

import numpy as np
import shapely.geometry as geo

import massivepy.spectrum as spec
import massivepy.constants as const

import utilities as utl


test_data = 'data/mitchell-rawcubes/QnovallfibNGC1600_log.fits'
x, y, pa = 67.9161, -5.0861, 15.00000  # degrees, degrees, degrees
nominal_const_fwhm = 4.5  # A
lower_trim = 3650
mask_threshold = 1000

# read Jenny's fiber datacube
cube = utl.fits_quickread(test_data)
spectra, noise, all_waves, coords, arcs = cube[0]
waves = all_waves[0, :]
valid = waves > lower_trim
spectra = spectra[:, valid]
noise = noise[:, valid]
waves = waves[valid]
ir = nominal_const_fwhm*np.ones(spectra.shape) # set fake ir
fiber_ids = np.arange(spectra.shape[0])
bad_data = np.absolute(spectra) > mask_threshold # mask bad pixels
# make SpectrumSet object
comments = {'object':'ngc1600',
            'instrument':'Mitchell Spectrograph (IFU)',
            'comment':'bundled data for software testing'}
specset = spec.SpectrumSet(spectra=spectra, bad_data=bad_data, noise=noise,
                           ir=ir, spectra_ids=fiber_ids, wavelengths=waves,
                           spectra_unit=const.flux_per_angstrom,
                           wavelength_unit=const.angstrom, comments=comments)

# get odd-numbered fibers from 2nd dither
num_in_dither = specset.num_spectra/3
dither2 = np.arange(num_in_dither, num_in_dither*2)
dither2_odd = dither2[dither2 % 2 == 1]
specsubset = specset.get_subset(dither2_odd)

# attempt normalize
specset_median = lambda s: 1.0/np.median(s.spectra, axis=1)
med_normedset = specset.get_normalized(specset_median)
delta_lambda = specset.spec_region[1] - specset.spec_region[0]
flux_normedset = specset.get_normalized(spec.SpectrumSet.compute_flux,
                                        delta_lambda)
# # test binning
binned = specset.collapse(id=0, weight_func=spec.SpectrumSet.compute_flux,
                          norm_func=spec.SpectrumSet.compute_flux,
                          norm_value=delta_lambda)
delta_lambda = binned.spec_region[1] - binned.spec_region[0]
binned = binned.get_normalized(norm_func=spec.SpectrumSet.compute_flux,
                               norm_value=delta_lambda)

s2n_fibers = specset.compute_mean_s2n()
s2n_binned = binned.compute_mean_s2n()

print "log:", specset.is_log_sampled()
print "linear:", specset.is_linear_sampled()
print "to linear"
specset.linear_resample()
print "now linear"
print "lin scale", specset.get_wavescale()
print "log re-sampling:"
print "num points", specset.waves.shape
specset.log_resample()
print "now log"
print "num points", specset.waves.shape
print "try to log again"
specset.log_resample()
print "try with halved step"
current_logscale = specset.get_logscale()
specset.log_resample(current_logscale/2.0)
print "num points", specset.waves.shape
print "back to linear"
specset.linear_resample()
print "now linear"
try:
    lin = specset.get_wavescale()
except Exception, msg:
    print msg
print "try to linear again"
specset.linear_resample()
print "num points", specset.waves.shape
print "lin scale", specset.get_wavescale()
