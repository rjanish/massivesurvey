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

import pickle

import numpy as np
import pandas as pd
import shapely.geometry as geo
import descartes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages


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

# empty list for outputs to plot
things_to_plot = []

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
    bin_type = input_params['bin_type']

    #New system for galaxy name, just taking from param file!!
    gal_name = os.path.basename(paramfile_path)[0:7]
    output_paths = {}
    output_paths['fits'] = os.path.join(destination_dir,
                                    "{}-{}.fits".format(gal_name,bin_type))
    output_paths['proc_cube'] = proc_cube_path
    output_paths['target_positions_path']=input_params['target_positions_path']
    things_to_plot.append(output_paths)
    if os.path.isfile(output_paths['fits']):
        if input_params['skip_rerun']=='yes':
            print '\nSkipping re-run of {}, plotting only'.format(gal_name)
            continue
        elif input_params['skip_rerun']=='no':
            print '\nRunning {} again, will overwrite output'.format(gal_name)
        else:
            raise Exception("skip_rerun must be yes or no")

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
    if bin_type=='unfolded':
        apf = functools.partial(binning.partition_quadparity,
                                major_axis=ma_bin, aspect_ratio=aspect_ratio)
    elif bin_type=='folded':
        apf = functools.partial(binning.partition_quadparity_folded,
                                major_axis=ma_bin, aspect_ratio=aspect_ratio)
    else:
        raise Exception('Bin type must be folded or unfolded, try again.')
    binning_func = functools.partial(binning.polar_threshold_binning,
                                     angle_partition_func=apf)
    binned = ifuset.s2n_fluxweighted_binning(get_bins=binning_func,
                                             threshold=s2n_threshold)
    print 'yaydone'
    grouped_ids, radial_bounds, angular_bounds = binned
    # results
    number_bins = len(grouped_ids)
    bin_ids = np.arange(number_bins, dtype=int) + 1  # bin 0 is full galaxy
    #Prep stuff for adding to new specset and metadata for each bin
    binned_data_shape = (number_bins, ifuset.spectrumset.num_samples)
    binned_data = {"spectra":np.zeros(binned_data_shape),
                   "bad_data":np.zeros(binned_data_shape),
                   "noise":np.zeros(binned_data_shape),
                   "ir":np.zeros(binned_data_shape),
                   "spectra_ids":bin_ids, # TO DO: add radial sorting
                   "wavelengths":ifuset.spectrumset.waves}
    bin_coords = np.zeros((number_bins, 4))  # flux weighted x, y, r, theta
    delta_lambda = (ifuset.spectrumset.spec_region[1] -
                    ifuset.spectrumset.spec_region[0])
    for bin_iter, fibers in enumerate(grouped_ids):
        bin_number = bin_iter + 1
        subset = ifuset.get_subset(fibers)
        binned = subset.spectrumset.collapse(
                                 weight_func=spec.SpectrumSet.compute_flux,
                                 norm_func=spec.SpectrumSet.compute_flux,
                                 norm_value=delta_lambda, id=bin_number)
        binned_data["spectra"][bin_iter, :] = binned.spectra
        binned_data["bad_data"][bin_iter, :] = binned.metaspectra["bad_data"]
        binned_data["noise"][bin_iter, :] = binned.metaspectra["noise"]
        binned_data["ir"][bin_iter, :] = binned.metaspectra["ir"]
        xs, ys = subset.coords.T
        fluxes = subset.spectrumset.compute_flux()
        bin_coords[bin_iter,:] = binning.calc_bin_center(xs,ys,fluxes,bin_type,
                                        ma=ma_bin,rmin=np.min(radial_bounds))
    spec_unit = ifuset.spectrumset.spec_unit
    wave_unit = ifuset.spectrumset.wave_unit
    binned_comments = ifuset.spectrumset.comments.copy()
    binned_comments["binning"] = "spectra have been spatially binned"
    binned_specset = spec.SpectrumSet(spectra_unit=spec_unit,
                                      wavelength_unit=wave_unit,
                                      comments=binned_comments,
                                      name=bin_type,
                                      **binned_data)
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
    output_base = os.path.join(destination_dir,
                               "{}-{}".format(ngc_name, bin_type))
    binned_data_path = "{}.fits".format(output_base)
    binned_specset.write_to_fits(binned_data_path)
    ###Here be pickle files, BEWARE
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
    ###End of pickle file territory
    print 'You may ignore the weird underflow error, it is not important.'

for data_paths in things_to_plot:
    basepath = data_paths['fits'][:-5]
    plot_path = "{}.pdf".format(basepath)
    binfiberpath = "{}_binfibers.p".format(basepath)
    bincoordpath = "{}_fluxcenters.p".format(basepath)
    proc_cube_path = data_paths['proc_cube']
    rad_path = "{}_radialbounds.p".format(basepath)
    gal_name = os.path.basename(basepath)[0:7]

    specset = spec.read_datacube(data_paths['fits'])
    #print specset.__dict__.keys()
    grouped_ids = pickle.load(open(binfiberpath,'r'))
    nbins = len(grouped_ids)
    ifuset = ifu.read_mitchell_datacube(proc_cube_path)
    fiber_coords = ifuset.coords.copy()
    fibersize = const.mitchell_fiber_radius.value #Assuming units match!
    bin_coords = pickle.load(open(bincoordpath,'r'))
    fiber_coords[:, 0] *= -1  # east-west reflect
    bin_coords[:, 0] *= -1
    squaremax = np.amax(np.abs(ifuset.coords)) + fibersize
    radial_bounds = pickle.load(open(rad_path,'r'))
    ang_paths = ["{}_angularbounds-{}.p".format(basepath,i) 
                 for i in range(len(radial_bounds))]
    angular_bounds = []
    for angpath in ang_paths:
        angular_bounds.append(pickle.load(open(angpath,'r')))

    target_positions = pd.read_csv(data_paths['target_positions_path'],
                                   comment='#', sep="[ \t]+",
                                   engine='python')
    gal_position = target_positions[target_positions.Name == gal_name]
    gal_pa = gal_position.PA_best.iat[0]
    ma_bin = np.pi/2 - np.deg2rad(gal_pa)
    ma_xy = np.pi/2 + np.deg2rad(gal_pa)

    ###Plotting begins!!!
    pdf = PdfPages(plot_path)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_axes([0.15,0.1,0.7,0.7])
    mycolors = ['b','g','c','m','r','y']
    used_fibers = []
    for n, fibers  in enumerate(grouped_ids):
        bin_color = mycolors[n % len(mycolors)]
        for fiber in fibers:
            used_fibers.append(fiber)
            ax.add_patch(patches.Circle(fiber_coords[fiber, :], fibersize,
                                facecolor=bin_color, zorder=0,
                                linewidth=0.25, alpha=0.8))
        #Plot flux-weighted bin centers (adjust size of stars for bin size)
        ms = 7.0 + 0.5*len(fibers)
        if ms > 20.0: ms = 20.0
        mew = 1.0 + 0.05*len(fibers)
        if mew > 2.0: mew = 2.0
        ax.plot(bin_coords[n][0],bin_coords[n][1],ls='',marker='*',
                mew=mew,ms=ms,mec='k',mfc=bin_color)
    # gray-out unbinned fibers
    for unused_fiber in range(fiber_coords.shape[0]):
        if unused_fiber not in used_fibers:
            ax.add_patch(patches.Circle(fiber_coords[unused_fiber, :],
                                        fibersize, facecolor='k', zorder=0,
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
    ax.plot([-rmax*1.1*np.cos(ma_xy), rmax*1.1*np.cos(ma_xy)],
            [-rmax*1.1*np.sin(ma_xy), rmax*1.1*np.sin(ma_xy)],
            linewidth=1.5, color='r')
    #These are hard coded in right now and that is bad!! fix it!!
    ax.set_title("binning - s/n = {}, Darc/Drad = {}"
                 "".format(s2n_threshold, aspect_ratio))
    ax.set_xlabel("arcsec")
    ax.set_ylabel("arcsec")
    ax.axis([-squaremax,squaremax,-squaremax,squaremax])
    pdf.savefig(fig)
    plt.close(fig)

    #Ok, now gonna plot each spectrum. For now I guess just jam them 
    # all onto one page to make my life easier, pretty it up later.
    fig = plt.figure(figsize=(6,0.5*nbins))
    ax = fig.add_axes([0.05,0.05,0.9,0.9])
    for ibin in range(nbins):
        ax.plot(specset.waves,specset.spectra[ibin,:]+ibin,c='k')
    ax.autoscale(tight=True)
    pdf.savefig(fig)
    plt.close(fig)
        

    pdf.close()
