

import os
import sys
import pickle

import numpy as np

from fileutils import re_filesearch
from mathutils import gausshermite_pdf


data_dir = sys.argv[1]
kin_results_dir = sys.argv[2]

bindata_filename = "ngc1600-bindata.p"
gh_params_pattern = (r"ngc1600_sublib_add0_mult3-bin\d{2}-"
                     r"4103.65_5684.44-gh_params.txt")


def round_by_step(num, step):
    excess = num % step
    if excess < step/2.0:
        rounded = num - excess
    else:
        rounded = num - excess + step
    return rounded


bindata_path = os.path.join(data_dir, bindata_filename)
with open(bindata_path, 'rb') as bindata_file:
    bindata = pickle.load(bindata_file)
bins, junk, bin_fluxes, fiber_fluxes, fiber_coords_equatorial = bindata
# This bindata is odd - the bin boundaries entry is not formatted
# correctly, and I am not sure why. It was made by Stephen, so perhaps
# he changed something, but regardless the bin bounds are unreadable.
# I must reconstruct them using the shapely plotting polygons.


# These coordinates are all in units of ra, cos(ra)*dec, need to rotate to
# coordinates with +x at 15 degrees E of N and +y at 105 degrees E of N.
# So, currently +x = E and +y = N, and we apply a reflection in x to get
# +x = W or (-90 E of N), +y = N (0 E of N) and then rotate clockwise
# by 90 + 15 degrees to get: +x = -(90 + 105) E of N= 15 E of N and
# +y = (0 + 105) E of N = 105 E of N.
pa = 15  # deg
angle = -np.deg2rad(90 + pa)  # - b/c below matrix assumes counter-clockwise
rotation = np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])
fiber_coords = np.zeros(fiber_coords_equatorial.shape)
for fiber_index, coords_eq in enumerate(fiber_coords_equatorial):
    coords_ma = coords_eq.copy()
    coords_ma[0] *= -1  # reflect x
    fiber_coords[fiber_index, :] = np.dot(rotation, coords_ma)

single_fibers = [f[0] for f, p in bins if p is None]
multifiber_bin_polys = [p for f, p in bins if p is not None]
multifiber_bin_fibers = [f for f, p in bins if p is not None]
num_bins = len(bins)
num_singles = len(single_fibers)
num_multi = len(multifiber_bin_polys)
bin_numbers = 1 + np.arange(num_bins)  # bin 0 is full galaxy spectrum

bin_centers = np.zeros((num_bins, 2))
bin_centers[:num_singles, :] = fiber_coords[single_fibers, :]
for multifiber_index, fibers in enumerate(multifiber_bin_fibers):
    bin_number = multifiber_index + num_singles
    coords = fiber_coords[fibers, :]
    below_ma = coords[:, 1] < 0
    coords[below_ma, 1] *= -1
    fluxes = fiber_fluxes[fibers]
    total_flux = np.sum(fluxes)
    for coord in xrange(2):
        center_of_light = np.sum(coords[:, coord]*fluxes)/total_flux
        bin_centers[bin_number, coord] = center_of_light
centers_header = """ngc1600 flux-weighted center of bins
the bin center is reported differently for single and multi fiber bins
single-fiber bins: the bin center is the fiber center, with no
  reflections nor foldings
multi-fiber bins: these bins are folded across the major axis, the center
  is given as the flux-weighted center of all fibers in the bin after
  reflecting all fibers to the +y side of the major axis
col 1: bin number
col 2: x, arcsec
col 3: y, arcsec
coordinate system:
origin corresponds to the galactic center
axis directions are: +x = 15 degrees E of N, +y = 105 degrees E of N
+x is aligned with the major axis
origin located at ra, dec = 67.9161 degrees, -5.0861 degrees
pa of major axis: 15.0 degrees E of N"""
centers_filename = "ngc1600-ppxf-bincenters.txt"
output_shape = np.array(bin_centers.shape)
output_shape[1] += 1
output = np.zeros(output_shape)
output[:, 0] = bin_numbers
output[:, 1:] = bin_centers
np.savetxt(centers_filename, output, header=centers_header,
           delimiter='  ', fmt=['%3d', '%8.3f', '%8.3f'])


bin_bounds = np.zeros((num_multi, 4))
all_edge_coords = []
for multi_index, poly in enumerate(multifiber_bin_polys):
    edge_coords_equatorial = np.array(poly.boundary.coords)
    # polygons are stored in equatorial coordinates, transform to MA coords
    edge_coords = np.zeros(edge_coords_equatorial.shape)
    for edgefiber_index, coords_eq in enumerate(edge_coords_equatorial):
        coords_ma = coords_eq.copy()
        coords_ma[0] *= -1
        edge_coords[edgefiber_index, :] = np.dot(rotation, coords_ma)
    all_edge_coords.append(edge_coords)
    # get boundaries
    r = np.sqrt(np.sum(edge_coords**2, axis=1))  # arcsec
    r = np.round(r, 3)
    theta = np.arctan2(edge_coords[:, 1], edge_coords[:, 0])  # rad
    theta = np.rad2deg(theta).astype(int)
        # This makes the reflection math below easier, but it will results in
        # occasional gaps of +- 1 degree in the angular bounds - it is assumed
        # that the bounds were originally set at some integer step greater
        # than 1 degree, and later rounding to integer multiples of this step
        # can be used to remove any gaps (step = 5 degrees is used below)
    # NOTE:
    # there is ambiguity in the ordering of the angles - we'll have +-
    # confusion when a set of angles crossed 0 or 180.  Since our binning
    # scheme is designed to have bin edges on the major axis, after
    # transforming to MA coordinates we have only the issue of bins that
    # start or stop on 180 or 0.  I'll pin these edge values to 180 or 0,
    # and then reflect all others to run between 0 and 180.
    minus_x_axis = (theta % 360) == 180
    theta[minus_x_axis] = 180
    plus_x_axis = (theta % 360) == 0
    theta[plus_x_axis] = 0
    theta[theta < 0] *= -1
        # I have check that the above works at the time of this writing,
        # but it requires that all angle boundaries are initially expressed
        # between -180 and 0, so it is NOT general.
    bin_bounds[multi_index, 0] = r.min()
    bin_bounds[multi_index, 1] = r.max()
    bin_bounds[multi_index, 2] = round_by_step(theta.min(), 5)
    bin_bounds[multi_index, 3] = round_by_step(theta.max(), 5)
bounds_header = """ngc1600 boundaries of multi-fiber bins
These bins are folded, and so their spacial footprint consists of two
regions that are reflections of each other across the major axis.
Only the regions in the +y half-plane are reported below.
col 1: bin number
col 2: inner radius, arcsec
col 3: outer radius, arcsec
col 4: inner angle, degrees
col 5: outer angle, degrees
radii are wrt galactic center
center located at ra, dec = 67.9161 degrees, -5.0861 degrees
angles are defined to be zero at the major axis and increasing towards E
pa of major axis is 15.0 degrees E of N"""
bounds_filename = "ngc1600-ppxf-binbounds.txt"
output_shape = np.array(bin_bounds.shape)
output_shape[1] += 1
output = np.zeros(output_shape)
output[:, 0] = bin_numbers[num_singles:]
output[:, 1:] = bin_bounds
np.savetxt(bounds_filename, output, header=bounds_header,
           delimiter='  ', fmt=['%3d', '%8.3f', '%8.3f', '%8.3f', '%8.3f'])


singlefiber_header = """ngc1600 single-fiber bins
col 1: bin number
col 2: x, arcsec
col 3: y, arcsec
coordinate system:
origin corresponds to the galactic center
axis directions are: +x = 15 degrees E of N, +y = 105 degrees E of N
+x is aligned with the major axis
origin located at ra, dec = 67.9161 degrees, -5.0861 degrees
pa of major axis: 15.0 degrees E of N"""
singles_filename = "ngc1600-ppxf-singlefibers.txt"
single_coords = fiber_coords[single_fibers, :]
output_shape = np.array(single_coords.shape)
output_shape[1] += 1
output = np.zeros(output_shape)
output[:, 0] = bin_numbers[:num_singles]
output[:, 1:] = single_coords
np.savetxt(singles_filename, output, header=singlefiber_header,
           delimiter='  ', fmt=['%3d', '%8.3f', '%8.3f'])

gh_param_paths = re_filesearch(gh_params_pattern, kin_results_dir)[0]
gh_param_paths.sort()
for bin_number, path in enumerate(gh_param_paths):
    gh_params = np.loadtxt(path)
    samples_name = "ngc1600-bin{:02d}-gh_params.txt".format(bin_number)
    np.savetxt(samples_name, gh_params.T, delimiter='  ',
               fmt=["%8.3f", "%6.2f", "%6.3f", "%6.3f", "%6.3f", "%6.3f"])