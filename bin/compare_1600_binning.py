"""
This script compare the bin layout and resulting kinematics for two
binnings of NGC 1600.

The first of these binning was made by Stephen (ADD DETAILS) and fit
by Ryan Janish in May 2015. This used binning code dating from August
2014 (CHECK).  The second was binning by Ryan Janish on May 20 2015,
using binning code dating from May 20 2015.
"""


import os
import pickle
import functools
import subprocess
import sys # temporary debug

import numpy as np
import descartes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import astropy.io.fits as fits

import utilities as utl
import plotting.geo_utils as geo_utils
import massivepy.constants as const
import massivepy.spectrum as spec
import massivepy.IFUspectrum as ifu
import massivepy.binning as binning


# target
x, y, pa = 67.9161, -5.0861, 15.00000  # degrees, degrees, degrees
ma_xy = np.pi/2 + np.deg2rad(pa)
    # pa is degrees E of N
    # for x=W=-E, y=N, ma above is radians counterclockwise from x
datacube = 'data/mitchell-cubes/QnovallfibNGC1600_log.fits'
# instrument
fiber_radius = const.mitchell_fiber_radius.value  # arcsec
nominal_const_fwhm = 4.5  # A
mask_threshold = 10**4
# binned 2014-08 locations
old_dir = 'old_results/ngc1600'
old_data_dir = 'data'
old_data_pattern = r"bin\d{2}_ngc1600.fits"
old_kins_filename = 'ngc1600mitchell-lib_optfull00-7-gh_params.txt'
old_errs_filename = 'ngc1600mitchell-lib_optfull00-7-gh_params_errors.txt'
old_binpickle_filename = 'ngc1600-bindata.p'
old_binbounds_filename = "ngc1600-ppxf-binbounds.txt"
old_results_dir = os.path.join(old_dir, 'results')
old_kins_path = os.path.join(old_results_dir, old_kins_filename)
old_errs_path = os.path.join(old_results_dir, old_errs_filename)
old_binpickle_path = os.path.join(old_dir, old_binpickle_filename)
old_binbounds_path = os.path.join(old_dir, old_binbounds_filename)
# new results location
new_results_dir = "ngc1600_bincomparison"
new_data_dir = os.path.join("ngc1600_bincomparison/data_ivar")
new_kins_filename = "ngc1600mitchell-201505compare-3900.0_5200.0-gh_params.txt"

# re-bin settings
aspect_ratio = 1.5
threshold = 20
step_size = None

# get binned 2014-08 results
old_kins = np.loadtxt(old_kins_path)
old_errs = np.loadtxt(old_kins_path)
with open(old_binpickle_path, 'rb') as old_binpickle_file:
    old_bindata = pickle.load(old_binpickle_file)
[old_binned_fibers, old_bincenters, old_binfluxes,
 old_fiberfluxes, old_fibercoords] = old_bindata
    # not positive, but pretty sure these fiber coords use x=ra, y=dec
old_binbounds = np.loadtxt(old_binbounds_path)
    # cols: bin_num, rin, rout, ang_start, ang_end
    # angles are given in degrees from major axis, increasing towards East
old_single_fiber_bins = [l for l, p in old_binned_fibers if len(l) == 1]
old_flat_binned_fibers = [f for l, p in old_binned_fibers for f in l]
old_unbinned_fibers = [f for f in np.arange(old_fiberfluxes.shape[0])
                       if f not in old_flat_binned_fibers]
old_datafiles_paths = utl.re_filesearch(old_data_pattern,
                                        os.path.join(old_dir, old_data_dir))[0]
old_datafiles_paths.sort()
old_binneddata = np.asarray([utl.fits_quickread(path)[0][0]
                             for path in old_datafiles_paths])
old_waves = old_binneddata[0, 2, :] # waves are uniform
old_spectra = old_binneddata[1:, 0, :] # skip full galaxy bin
old_noise = old_binneddata[1:, 1, :]
old_mask = old_binneddata[1:, 3, :].astype(bool)

# re-bin now
# read Jenny's fiber datacube
cube = utl.fits_quickread(datacube)
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
# make ifu object and bin it
comments = {'object':'ngc1600',
            'instrument':'Mitchell Spectrograph (IFU)',
            'comment':'bundled data for software testing'}
ifuset = ifu.IFUspectrum(spectra=spectra, bad_data=bad_data, noise=noise,
                         ir=ir, spectra_ids=fiber_ids, wavelengths=waves,
                         spectra_unit=const.flux_per_angstrom,
                         wavelength_unit=const.angstrom, comments=comments,
                         coords=coords, coords_unit=const.arcsec,
                         linear_scale=fiber_radius, footprint=fiber_circle)
ma_binning = np.pi/2 - np.deg2rad(pa)
unfolded_1600 = functools.partial(binning.partition_quadparity,
                                  major_axis=ma_binning,
                                  aspect_ratio=aspect_ratio)
folded_1600 = functools.partial(binning.partition_quadparity_folded,
                                major_axis=ma_binning,
                                aspect_ratio=aspect_ratio)
binning_func = functools.partial(binning.polar_threshold_binning,
                                 step_size=step_size,
                                 angle_partition_func=folded_1600)
# combine_func = functools.partial(spec.SpectrumSet.collapse, id=0,
#                                  weight_func=spec.SpectrumSet.compute_flux)
ivar = lambda s: s.metaspectra['noise']**(-2)
combine_func = functools.partial(spec.SpectrumSet.collapse, id=0,
                                 weight_func=ivar)
binned = ifuset.s2n_spacial_binning(binning_func=binning_func,
                                    combine_func=combine_func,
                                    threshold=threshold)
grouped_ids, radial_bounds, angular_bounds = binned
    # angles here are cc from x-axis, with x=E y=N, ie E to N
new_single_fiber_bins = [l for l in grouped_ids if len(l) == 1]
new_flat_binned_fibers = [f for l in grouped_ids for f in l]
new_unbinned_fibers = [f for f in fiber_ids if f not in new_flat_binned_fibers]
# make binned spectra
new_binned_specsets = [combine_func(ifuset.spectrumset.get_subset(f))
                       for f in grouped_ids]
new_waves = new_binned_specsets[0].waves # waves are uniform
new_spectra = np.asarray([s.spectra[0] for s in new_binned_specsets])
new_noise = np.asarray([s.metaspectra['noise'][0] for s in new_binned_specsets])
new_mask = np.asarray([s.metaspectra['bad_data'][0] for s in new_binned_specsets])
for bin_iter in xrange(len(grouped_ids)):
    bin_num = bin_iter + 1
    data_package = np.asarray([new_spectra[bin_iter],
                               new_noise[bin_iter], new_waves,
                               new_mask[bin_iter].astype(float)])
    path = os.path.join(new_data_dir, "bin{:02d}_ngc1600_201505.fits".format(bin_num))
    fits.writeto(path, data_package, clobber=True)

print "data saved"
sys.exit()

# get new kins
new_kins = np.loadtxt(os.path.join(new_results_dir, new_kins_filename))
figs = []
for moment in xrange(6):
    fig, ax = plt.subplots()
    ax.plot(old_kins[moment, 1:], linestyle='', marker='o',
            alpha=0.6, label='August 2014')
    ax.plot(new_kins[moment, :], linestyle='', marker='o',
            alpha=0.6, label='May 20 2015')
    ax.set_title("param {}".format(moment))
    ax.legend(loc='best', fontsize=12)
    figs.append("ngc1600_Aug14May15bincompare-{}.pdf".format(moment))
    fig.savefig("ngc1600_Aug14May15bincompare-{}.pdf".format(moment))
    plt.close(fig)
cmd = "pdfunite {} {}".format(" ".join(figs), "ngc1600_Aug14May15bincompare.pdf")
subprocess.call(cmd, shell=True)


# print compares
print "August 2014 bins:"
print "{} total number of bins".format(len(old_binned_fibers))
print "{} single-fiber bins".format(len(old_single_fiber_bins))
print "{} un-binned outer fibers".format(len(old_unbinned_fibers))
print "innermost multi-fiber bin boundary: {:4.1f}".format(old_binbounds[0][1])
print "outermost multi-fiber bin boundary: {:4.1f}".format(old_binbounds[-1][2])
print "May 20 2015 bins:"
print "{} total number of bins".format(len(grouped_ids))
print "{} single-fiber bins".format(len(new_single_fiber_bins))
print "{} un-binned outer fibers".format(len(new_unbinned_fibers))
print "innermost multi-fiber bin boundary: {:4.1f}".format(radial_bounds[0][0])
print "outermost multi-fiber bin boundary: {:4.1f}".format(radial_bounds[-1][1])
# plots
fig, ax = plt.subplots()
old_fiber_xy = old_fibercoords.copy()
old_fiber_xy[:, 0] *= -1  # east-west reflect: x=W=-E, y=N
# plot grayed old fibers
for fiber in range(old_fiber_xy.shape[0]):
    ax.add_patch(patches.Circle(old_fiber_xy[fiber, :], fiber_radius,
                                facecolor='k', zorder=0,
                                linewidth=0.25, alpha=0.15))
# plot bin outlines, old
for [bin_num, rmin, rmax, amin, amax] in old_binbounds:
    amin_xy = amin + np.rad2deg(ma_xy) # angles start as degrees from MA to
    amax_xy = amax + np.rad2deg(ma_xy) # E, are now degrees cc-wise from x
    bin_poly = geo_utils.polar_box(rmin, rmax, amin_xy, amax_xy)
    ax.add_patch(descartes.PolygonPatch(bin_poly, facecolor='none',
                                        edgecolor='k', linestyle='solid',
                                        linewidth=1.5))
    # only E side of MA is listed, plot other side as well
    amin_xy_r = -amax + np.rad2deg(ma_xy)
    amax_xy_r = -amin + np.rad2deg(ma_xy)
    bin_poly_r = geo_utils.polar_box(rmin, rmax, amin_xy_r, amax_xy_r)
    ax.add_patch(descartes.PolygonPatch(bin_poly_r, facecolor='none',
                                        edgecolor='k', linestyle='solid',
                                        linewidth=1.5))
ax.plot([], [], color='k', linestyle='solid', linewidth=1.5,
        label='bins August 2014')
# plot bins
new_fiber_xy = ifuset.coords.copy()
new_fiber_xy[:, 0] *= -1  # east-west reflect
# plot grayed new fibers
for fiber in range(new_fiber_xy.shape[0]):
    ax.add_patch(patches.Circle(new_fiber_xy[fiber, :], fiber_radius,
                                facecolor='k', zorder=0,
                                linewidth=0.25, alpha=0.15))
# plot bin outlines
for n, (rmin, rmax) in enumerate(radial_bounds):
    for angular_bins in angular_bounds[n]:
        for amin_NofE, amax_NofE in angular_bins:
            amin_xy = np.pi - amax_NofE
            amax_xy = np.pi - amin_NofE
            bin_poly = geo_utils.polar_box(rmin, rmax, np.rad2deg(amin_xy),
                                           np.rad2deg(amax_xy))
            ax.add_patch(descartes.PolygonPatch(bin_poly, facecolor='none',
                                                edgecolor='r', linewidth=1.5,
                                                linestyle='solid'))
ax.plot([], [], color='r', linestyle='solid', linewidth=1.5,
        label='bins May 20 2015')
# add major axis
ax.plot([-rmax*1.3*np.cos(ma_xy), rmax*1.3*np.cos(ma_xy)],
        [-rmax*1.3*np.sin(ma_xy), rmax*1.3*np.sin(ma_xy)],
        linewidth=1.5, color='k', linestyle=':', label='major axis')
ax.autoscale_view()
ax.set_aspect('equal')
ax.legend(loc='best', fontsize=12)

plt.show()






