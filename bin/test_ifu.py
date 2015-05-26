"""
Test basic functionality of IFUspectrum
"""


import numpy as np
import shapely.geometry as geo

import massivepy.IFUspectrum as ifu
import massivepy.constants as const

import utilities as utl


test_data = 'data/mitchell-cubes/QnovallfibNGC1600_log.fits'
x, y, pa = 67.9161, -5.0861, 15.00000  # degrees, degrees, degrees
nominal_const_fwhm = 4.5  # A
mask_threshold = 10**4

# read Jenny's fiber datacube
cube = utl.fits_quickread(test_data)
spectra, noise, all_waves, coords, arcs = cube[0]
waves = all_waves[0, :]
ir = nominal_const_fwhm*np.ones(spectra.shape) # set fake ir
fiber_ids = np.arange(spectra.shape[0])
bad_data = np.absolute(spectra) > mask_threshold # mask bad pixels
# adjust coords to projected, scaled distance
coords[:, 0] = coords[:, 0] - x
coords[:, 1] = coords[:, 1] - y
coords[:, 0] = coords[:, 0]*np.cos(np.deg2rad(y))
coords = coords*60*60  # to arcsec
# fiber footprint
fiber_radius = const.mitchell_fiber_radius.value  # arcsec
fiber_circle = lambda center: geo.Point(center).buffer(fiber_radius)
# make ifu object
comments = {'object':'ngc1600',
            'instrument':'Mitchell Spectrograph (IFU)',
            'comment':'bundled data for software testing'}
ifuset = ifu.IFUspectrum(spectra=spectra, bad_data=bad_data, noise=noise,
                         ir=ir, spectra_ids=fiber_ids, wavelengths=waves,
                         spectra_unit=const.flux_per_angstrom,
                         wavelength_unit=const.angstrom, comments=comments,
                         coords=coords, coords_unit=const.arcsec,
                         linear_scale=fiber_radius, footprint=fiber_circle)
ifuset = ifu.IFUspectrum(spectra=spectra, bad_data=bad_data,
                            noise=np.zeros(spectra.shape),
                         ir=ir, spectra_ids=fiber_ids, wavelengths=waves,
                         spectra_unit=const.flux_per_angstrom,
                         wavelength_unit=const.angstrom, comments=comments,
                         coords=coords, coords_unit=const.arcsec,
                         linear_scale=fiber_radius, footprint=fiber_circle)

# get odd-numbered fibers from 2nd dither
num_in_dither = ifuset.spectrumset.num_spectra/3
dither2 = np.arange(num_in_dither, num_in_dither*2)
dither2_odd = dither2[dither2 % 2 == 1]
ifusubset = ifuset.get_subset(dither2_odd)
