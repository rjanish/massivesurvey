

import sys
import os
import pickle
from functools import partial

import numpy as np
from scipy.integrate import trapz
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from descartes import PolygonPatch
from astropy.io import fits

from geo_utils import polar_box
from fileutils import read_dict_file
from polar_binning import (construct_polar_bins,
    unfolded_partitioner, inverse_variance, median_normalize)


cube_path = sys.argv[1]
arc_path = sys.argv[2]
target_info_path = sys.argv[3]
output_dir = sys.argv[4]
try:
    jeremy_data = (sys.argv[5][:2] == '-j')
    if jeremy_data:
        divider = float(sys.argv[5][2:])
    else:
        divider = None
except:
    jeremy_data = False
    divider = None

target_info = read_dict_file(target_info_path)
center = target_info['center_ra'], target_info['center_dec']  # degrees
pa = target_info['pa']  # degrees E of N
fiber_radius = target_info["fiber_radius"]  # arcsec
ma = np.pi/2.0 - np.deg2rad(pa)  # radians N of W
data_cube = fits.open(cube_path)
spectra, noise, wavelengths, coords, arcs = [hdu.data for hdu in data_cube]
data_cube.close()
mask = (np.absolute(spectra) > 10**4) | (np.absolute(spectra) > 10**4)
fluxes = np.zeros(spectra.shape[0])
for fiber_index, (fiber_spec, fiber_mask) in enumerate(zip(spectra, mask)):
    flux = trapz(fiber_spec[~fiber_mask],
                 wavelengths[fiber_index, ~fiber_mask])
    fluxes[fiber_index] = flux
# adjust coords to projected, scaled distance
coords[:, 0] = coords[:, 0] - center[0]
coords[:, 1] = coords[:, 1] - center[1]
coords[:, 0] = coords[:, 0]*np.cos(np.deg2rad(center[1]))
coords = coords*60*60  # arcsec
with open(arc_path, 'rb') as arc_file:
    line_centers, fwhm = pickle.load(arc_file)

if jeremy_data:
    distant = np.sqrt(np.sum(coords**2, axis=1)) > divider
    spectra_distant = spectra[distant, :]
    noise_distant = noise[distant, :]
    mask_distant = mask[distant, :]
    coords_distant = coords[distant, :]
    fluxes_distant = fluxes[distant]
    spectra = spectra[~distant, :]
    noise = noise[~distant, :]
    mask = mask[~distant, :]
    coords = coords[~distant, :]
    fluxes = fluxes[~distant]
    # bin distant fibers
    distant_bin_unorm = inverse_variance(spectra_distant, noise_distant,
                                         mask=mask_distant)
    distant_bin_norm = median_normalize(*distant_bin_unorm)
    distant_binned_data = np.array([distant_bin_norm[0], distant_bin_norm[1],
                                    wavelengths[0,:],  distant_bin_unorm[2]])
    good_fibers = np.all(np.isfinite(fwhm[distant]), axis=1)
    weights = fluxes_distant[good_fibers]/np.sum(fluxes_distant[good_fibers])
    centers = line_centers[distant][good_fibers]
    widths = fwhm[distant][good_fibers]
    binned_centers = np.sum(centers.T*weights, axis=1)
    binned_fwhm = np.sum(widths.T*weights, axis=1)
    distant_arcs = np.array([binned_centers, binned_fwhm]).T

# bin central fibers
partitoner = partial(unfolded_partitioner, aspect_ratio=1.5)
binned_output = construct_polar_bins(spectra, noise, mask, coords,
                                     major_axis=ma, step_size=fiber_radius,
                                     s2n_limit=20, fiber_radius=fiber_radius,
                                     angular_paritioner=partitoner)
binned_fibers, binned_data, annular_sets, single_fiber_radius = binned_output

bins = []
weighted_position = []
bin_flux = []
single_fibers = [fibers[0] for fibers in binned_fibers if len(fibers) == 1]
num_single_fibers = len(single_fibers)
for bin_index, fiber in enumerate(single_fibers):
    bins.append([[fiber], None])
    weighted_position.append(coords[fiber, :])
    bin_flux.append(fluxes[fiber])
for radial_index, ((rmin, rmax), angle_bounds) in enumerate(annular_sets):
    for angular_index, (amin, amax) in enumerate(angle_bounds):
        bin_index += 1
        bin_poly = polar_box(rmin, rmax, np.rad2deg(amin), np.rad2deg(amax))
        fibers = binned_fibers[bin_index]
        bins.append([fibers, bin_poly])
        total_flux = np.sum(fluxes[fibers])
        weights = fluxes[fibers]/total_flux
        bin_center = np.sum(coords[fibers, :].T*weights, axis=1)
        weighted_position.append(bin_center)
        bin_flux.append(total_flux)
binning_pickle_output = [bins, weighted_position, bin_flux, fluxes, coords]

# bin central fibers' arcs
binned_arcs = []
for n, fibers  in enumerate(binned_fibers):
    fibers = np.asarray(fibers)
    good_fibers = np.all(np.isfinite(fwhm[fibers]), axis=1)
    fibers = fibers[good_fibers]
    centers = line_centers[fibers]
    widths = fwhm[fibers]
    weights = fluxes[fibers]/np.sum(fluxes[fibers])
    binned_centers = np.sum(centers.T*weights, axis=1)
    binned_fwhm = np.sum(widths.T*weights, axis=1)
    binned_arcs.append(np.array([binned_centers, binned_fwhm]).T)
binned_arcs = np.array(binned_arcs)

# bin full central field
full_bin_unnorm = inverse_variance(spectra, noise, mask=mask)
full_bin_norm = median_normalize(*full_bin_unnorm)
full_binned_data = np.array([full_bin_norm[0], full_bin_norm[1],
                             wavelengths[0, :], full_bin_unnorm[2]])
all_central_fibers = [f for b in binned_fibers for f in b
                      if np.all(np.isfinite(fwhm[f]))]
weights = fluxes[all_central_fibers]/np.sum(fluxes[all_central_fibers])
binned_centers = np.sum(line_centers[all_central_fibers].T*weights, axis=1)
binned_fwhm = np.sum(fwhm[all_central_fibers].T*weights, axis=1)
full_arcs = np.array([binned_centers, binned_fwhm]).T

# plot binning
coords[:, 0] *= -1
# plots - each fiber colored by bin membership
colors = ['b', 'g', 'r', 'c', 'm']
used_fibers = []
fig, ax = plt.subplots()
for n, fibers  in enumerate(binned_fibers):
    # fibers_in_bins is a list of lists of fibers in each bin
    bin_color = colors[n % len(colors)]
    for fiber in fibers:
        used_fibers.append(fiber)
        ax.add_patch(Circle(coords[fiber, :], fiber_radius,
                            facecolor=bin_color, zorder=0,
                            linewidth=0.25, alpha=0.8))
    ax.set_aspect('equal')
# gray-out unbinned fibers
for unused_fiber in range(coords.shape[0]):
    if unused_fiber not in used_fibers:
        ax.add_patch(Circle(coords[unused_fiber, :], fiber_radius,
                            facecolor='k', zorder=0,
                            linewidth=0.25, alpha=0.3))
# plot bin outlines
for n, ((rmin, rmax), angle_bounds) in enumerate(annular_sets):
    for m, (amin, amax) in enumerate(angle_bounds):
        num = n + m
        amin_refl = np.pi - amin
        amax_refl = np.pi - amax
        bin_poly = polar_box(rmin, rmax, np.rad2deg(amin_refl),
                             np.rad2deg(amax_refl))
        ax.add_patch(PolygonPatch(bin_poly, facecolor='none',
                                  linestyle='solid', linewidth=1.5))
ax.autoscale_view()
plt.show()
plt.close(fig)

# output binned data
for bin_index, data in enumerate(binned_data):
    bin_number = bin_index + 1
    to_save = np.array([data[0], data[1], wavelengths[0, :], data[2]])
    filename = 'bin{:02d}_ngc1600jeremy.fits'.format(bin_number)
    arc_filename = 'bin{:02d}_ngc1600jeremy_ir.txt'.format(bin_number)
    path = os.path.join(output_dir, filename)
    arc_path = os.path.join(output_dir, arc_filename)
    np.savetxt(arc_path, binned_arcs[bin_index], delimiter='  ')
    fits.writeto(path, to_save, clobber=True)
pickle_path = os.path.join(output_dir, "ngc1600jereme-bindata.p")
with open(pickle_path, 'wb') as pkl_file:
    pickle.dump(binning_pickle_output, pkl_file)

full_filename = "bin00_ngc1600jeremy.fits"
full_arc_filename = "bin00_ngc1600jeremy_ir.txt"
full_path = os.path.join(output_dir, full_filename)
full_arc_path = os.path.join(output_dir, full_arc_filename)
np.savetxt(full_arc_path, full_arcs, delimiter='  ')
fits.writeto(full_path, full_binned_data, clobber=True)

distant_filename = "bindistant_ngc1600jeremy.fits"
distant_arc_filename = "bindistant_ngc1600jeremy_ir.txt"
distant_path = os.path.join(output_dir, distant_filename)
distant_arc_path = os.path.join(output_dir, distant_arc_filename)
np.savetxt(distant_arc_path, distant_arcs, delimiter='  ')
fits.writeto(distant_path, distant_binned_data, clobber=True)