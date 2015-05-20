"""
Test basic functionality of IFUspectrum
"""


import functools

import numpy as np
import shapely.geometry as geo
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import descartes  # replace with plotting library

import massivepy.spectrum as spec
import massivepy.IFUspectrum as ifu
import massivepy.binning as binning
import massivepy.constants as const
import utilities as utl
import plotting.geo_utils as geo_utils  # replace with plotting library


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
# do binning
ma = np.pi/2 - np.deg2rad(pa)
unfolded_1600 = functools.partial(binning.partition_quadparity,
                                  major_axis=ma, aspect_ratio=2)
binning_func = functools.partial(binning.polar_threshold_binning,
                                 step_size=fiber_radius,
                                 angle_partition_func=unfolded_1600)
combine_func = functools.partial(spec.SpectrumSet.collapse, id=0,
                                 weight_func=spec.SpectrumSet.compute_flux)
binned = ifuset.s2n_spacial_binning(binning_func=binning_func,
                                    combine_func=combine_func,
                                    threshold=20/np.sqrt(2))
grouped_ids, radial_bounds, angular_bounds = binned
# results
single_fiber_bins = [l for l in grouped_ids if len(l) == 1]
flat_binned_fibers = [f for l in grouped_ids for f in l]
unbinned_fibers = [f for f in fiber_ids if f not in flat_binned_fibers]
print "{} total number of bins".format(len(grouped_ids))
print "{} single-fiber bins".format(len(single_fiber_bins))
print "{} un-binned outer fibers".format(len(unbinned_fibers))
print "multi-fiber bin layout:"
for iter, [(rin, rout), angles] in enumerate(zip(radial_bounds, angular_bounds)):
    print ("  {:2d}: radius {:4.1f} to {:4.1f}, {} angular bins"
           "".format(iter + 1, rin, rout, len(angles)))
# plot bins
fiber_coords = ifuset.coords.copy()
fiber_coords[:, 0] *= -1  # east-west reflect
# plots - each fiber colored by bin membership
colors = ['b', 'g', 'r', 'c', 'm']
used_fibers = []
fig, ax = plt.subplots()
for n, fibers  in enumerate(grouped_ids):
    # fibers_in_bins is a list of lists of fibers in each bin
    bin_color = colors[n % len(colors)]
    for fiber in fibers:
        used_fibers.append(fiber)
        ax.add_patch(patches.Circle(fiber_coords[fiber, :], fiber_radius,
                            facecolor=bin_color, zorder=0,
                            linewidth=0.25, alpha=0.8))
    ax.set_aspect('equal')
# gray-out unbinned fibers
for unused_fiber in range(fiber_coords.shape[0]):
    if unused_fiber not in used_fibers:
        ax.add_patch(patches.Circle(fiber_coords[unused_fiber, :],
                                    fiber_radius, facecolor='k', zorder=0,
                                    linewidth=0.25, alpha=0.3))
# plot bin outlines
for n, (rmin, rmax) in enumerate(radial_bounds):
    for angular_bins in angular_bounds[n]:
        for amin, amax in angular_bins:
            bin_poly = geo_utils.polar_box(rmin, rmax, np.rad2deg(amin),
                                           np.rad2deg(amax))
            ax.add_patch(descartes.PolygonPatch(bin_poly, facecolor='none',
                                                linestyle='solid',
                                                linewidth=1.5))
ax.add_artist(patches.Circle((0, 0), radial_bounds[0][0], edgecolor='k',
              facecolor='none'))
ax.plot([-rmax*1.1*np.cos(ma), rmax*1.1*np.cos(ma)],
        [-rmax*1.1*np.sin(ma), rmax*1.1*np.sin(ma)],
        linewidth=1.5, color='r')
ax.autoscale_view()
ax.set_aspect('equal')

plt.show()
