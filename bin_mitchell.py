"""
Construct polar, S/N threshold binning of Mitchell IFU fibers.

input:
  takes one command line argument, a path to the input parameter text file
  bin_mitchell_params_example.txt is an example
  can take multiple parameter files if you want to process multiple galaxies
  (give one param file per galaxy)

output:
  One binned datacube per galaxy. (One file contains all bins.)
  A bunch of other stuff that will be cleaned up at some point
"""


import os
import re
import argparse
import functools

import numpy as np
import pandas as pd
import shapely.geometry as geo
import descartes
import matplotlib.pyplot as plt
import matplotlib.patches as patches


import utilities as utl
import massivepy.constants as const
import massivepy.IFUspectrum as ifu
import massivepy.spectrum as spec
import massivepy.binning as binning
import plotting.geo_utils as geo_utils


# get cmd line arguments
parser = argparse.ArgumentParser(description=__doc__,
                formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("paramfiles", nargs='*', type=str,
                    help="path(s) to input parameter file(s)")
args = parser.parse_args()
all_paramfile_paths = args.paramfiles


for paramfile_path in all_paramfile_paths:
    # parse input parameter file
    input_params = utl.read_dict_file(paramfile_path)
    proc_cube_path = input_params['proc_mitchell_cube']
    if ((not os.path.isfile(proc_cube_path)) 
        or (os.path.splitext(proc_cube_path)[-1] != ".fits")):
        raise ValueError("Invalid raw datacube path {}, "
                         "must be .fits file".format(proc_cube_path))
    aspect_ratio = input_params['aspect_ratio']
    s2n_threshold = input_params['s2n_threshold']
    target_positions = pd.read_csv(input_params['target_positions_path'],
                                   comment='#', sep="[ \t]+",
                                   engine='python')
    destination_dir = input_params['destination_dir']
    if not os.path.isdir(destination_dir):
        raise ValueError("Invalid destination dir {}".format(destination_dir))

    # get bin layout
    ifuset = ifu.read_mitchell_datacube(proc_cube_path)
    ngc_match = re.search(const.re_ngc, proc_cube_path)
    if ngc_match is None:
        msg = "No galaxy name found for path {}".format(proc_cube_path)
        raise RuntimeError(msg)
    else:
        ngc_num = ngc_match.groups()[0]
    ngc_name = "NGC{}".format(ngc_num)
    print "\n{}".format(ngc_name)
    print "  binning..."
    gal_position = target_positions[target_positions.Name == ngc_name]
    gal_pa = gal_position.PA_best.iat[0]
    ma_bin = np.pi/2 - np.deg2rad(gal_pa)
    ma_xy = np.pi/2 + np.deg2rad(gal_pa)
    fiber_radius = const.mitchell_fiber_radius.value
    folded = functools.partial(binning.partition_quadparity_folded,
                               major_axis=ma_bin, aspect_ratio=aspect_ratio)
    binning_func = functools.partial(binning.polar_threshold_binning,
                                     angle_partition_func=folded)
    binned = ifuset.s2n_fluxweighted_binning(get_bins=binning_func,
                                             threshold=s2n_threshold)
    grouped_ids, radial_bounds, angular_bounds = binned
    # results
    number_bins = len(grouped_ids)
    bin_ids = np.arange(number_bins, dtype=int) + 1  # bin 0 is full galaxy
    binned_data_shape = (number_bins, ifuset.spectrumset.num_samples)
    binned_data = {"spectra":np.zeros(binned_data_shape),
                   "bad_data":np.zeros(binned_data_shape),
                   "noise":np.zeros(binned_data_shape),
                   "ir":np.zeros(binned_data_shape),
                   "spectra_ids":bin_ids, # TO DO: add radial sorting
                   "wavelengths":ifuset.spectrumset.waves}
    bin_coords = np.zeros((number_bins, 4))  # flux weighted x, y, r, theta
    for bin_iter, fibers in enumerate(grouped_ids):
        bin_number = bin_iter + 1
        subset = ifuset.get_subset(fibers)
        delta_lambda = (ifuset.spectrumset.spec_region[1] -
                        ifuset.spectrumset.spec_region[0])
        binned = subset.spectrumset.collapse(
                                 weight_func=spec.SpectrumSet.compute_flux,
                                 norm_func=spec.SpectrumSet.compute_flux,
                                 norm_value=delta_lambda, id=bin_number)
        binned_data["spectra"][bin_iter, :] = binned.spectra
        binned_data["bad_data"][bin_iter, :] = binned.metaspectra["bad_data"]
        binned_data["noise"][bin_iter, :] = binned.metaspectra["noise"]
        binned_data["ir"][bin_iter, :] = binned.metaspectra["ir"]
        binned_comments = subset.spectrumset.comments.copy()
        binned_comments["binning"] = "spectra have been spatially binned"
        bindesc = "polar_folded_s2n20"
        spec_unit = spectra_unit=subset.spectrumset.spec_unit
        wave_unit = wavelength_unit=subset.spectrumset.wave_unit
        binned_specset = spec.SpectrumSet(spectra_unit=spec_unit,
                                          wavelength_unit=wave_unit,
                                          comments=binned_comments,
                                          name=bindesc, **binned_data)
        bin_xs, bin_ys = subset.coords.T
        fluxes = subset.spectrumset.compute_flux()
        total_flux = fluxes.sum()
        x_com = np.sum(bin_xs*fluxes)/total_flux
        y_com = np.sum(bin_ys*fluxes)/total_flux
        r_com = np.sqrt(x_com**2 + y_com**2)
        th_com = np.arctan2(y_com, x_com)
        bin_coords[bin_iter, :] = np.asarray([x_com, y_com, r_com, th_com])
    fiber_ids = ifuset.spectrumset.ids
    single_fiber_bins = [l for l in grouped_ids if len(l) == 1]
    flat_binned_fibers = [f for l in grouped_ids for f in l]
    unbinned_fibers = [f for f in fiber_ids if f not in flat_binned_fibers]
    # output
    print "  {} total number of bins".format(len(grouped_ids))
    print "  {} single-fiber bins".format(len(single_fiber_bins))
    print "  {} un-binned outer fibers".format(len(unbinned_fibers))
    print "  multi-fiber layout:"
    for iter, [(rin, rout), angles] in enumerate(zip(radial_bounds,
                                                     angular_bounds)):
        print ("   {:2d}: radius {:4.1f} to {:4.1f}, {} angular bins"
               "".format(iter + 1, rin, rout, len(angles)))
    output_base = os.path.join(destination_dir, "{}-{}".format(ngc_name, bindesc))
    binned_data_path = "{}.fits".format(output_base)
    binned_specset.write_to_fits(binned_data_path)
    binned_path = "{}_binfibers.p".format(output_base)
    utl.save_pickle(grouped_ids, binned_path)
    unbinned_path = "{}_unbinnedfibers.p".format(output_base)
    utl.save_pickle(unbinned_fibers, unbinned_path)
    rad_path = "{}_radialbounds.p".format(output_base)
    utl.save_pickle(radial_bounds, rad_path)
    coord_path = "{}_fluxcenters.p".format(output_base)
    utl.save_pickle(bin_coords, coord_path)
    for anulus_iter, angle_bounds in enumerate(angular_bounds):
        angle_path = "{}_angularbounds-{}.p".format(output_base, anulus_iter)
        utl.save_pickle(angle_bounds, angle_path)
    # plot bins
    # TO DO: move this to a MASSVIE plotting module
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
            for amin_NofE, amax_NofE in angular_bins:
                amin_xy = np.pi - amax_NofE
                amax_xy = np.pi - amin_NofE
                bin_poly = geo_utils.polar_box(rmin, rmax,
                                               np.rad2deg(amin_xy),
                                               np.rad2deg(amax_xy))
                ax.add_patch(descartes.PolygonPatch(bin_poly, facecolor='none',
                                                    linestyle='solid',
                                                    linewidth=1.5))
    ax.add_artist(patches.Circle((0, 0), radial_bounds[0][0], edgecolor='k',
                  facecolor='none'))
    ax.plot([-rmax*1.1*np.cos(ma_xy), rmax*1.1*np.cos(ma_xy)],
            [-rmax*1.1*np.sin(ma_xy), rmax*1.1*np.sin(ma_xy)],
            linewidth=1.5, color='r')
    ax.set_title("{} polar folded binning - s/n = {}, Darc/Drad = {}"
                 "".format(ngc_name, s2n_threshold, aspect_ratio))
    ax.set_xlabel("arcsec")
    ax.set_ylabel("arcsec")
    ax.autoscale_view()
    ax.set_aspect('equal')
    plot_path = "{}-binoutlines.pdf".format(output_base)
    fig.savefig(plot_path)
    plt.close(fig)

print 'You may ignore the weird underflow error, it is not important.'
