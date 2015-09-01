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
import time

import pickle

import numpy as np
import pandas as pd
import shapely.geometry as geo
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages


import utilities as utl
import massivepy.constants as const
import massivepy.IFUspectrum as ifu
import massivepy.spectrum as spec
import massivepy.binning as binning
import massivepy.io as mpio
from massivepy.plot_s2 import plot_s2_bin_mitchell


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
    print '\n\n====================================='
    # parse input parameter file
    output_dir, gal_name = mpio.parse_paramfile_path(paramfile_path)
    input_params = utl.read_dict_file(paramfile_path)
    raw_cube_path = input_params['raw_mitchell_cube']
    bad_fibers_path = input_params['bad_fibers_path']
    ir_path = input_params['ir_path']
    check = mpio.pathcheck([raw_cube_path,bad_fibers_path,ir_path],
                           ['.fits','.txt','.txt'],gal_name)
    targets_path = input_params['target_positions_path']
    if not os.path.isfile(targets_path):
        print "File {} does not exist".format(targets_path)
        check = False
    if not check:
        print 'Something is wrong with the input paths for {}'.format(gal_name)
        print 'Skipping to next galaxy.'
        continue
    run_name = input_params['run_name']
    aspect_ratio = input_params['aspect_ratio']
    s2n_threshold = input_params['s2n_threshold']
    bin_type = input_params['bin_type']
    crop_region = [input_params['crop_min'], input_params['crop_max']]
    fullbin_radius = input_params['fullbin_radius']

    # construct output file names
    output_path_maker = lambda f,ext: os.path.join(output_dir,
                "{}-s2-{}-{}.{}".format(gal_name,run_name,f,ext))
    binspectra_path = output_path_maker('binspectra','fits')
    fullbin_path = output_path_maker('fullgalaxy','fits')
    bininfo_path = output_path_maker('bininfo','txt')
    fiberinfo_path = output_path_maker('fiberinfo','txt')
    plot_path = output_path_maker('binmaps','pdf')
    # save relevant info for plotting to a dict
    plot_info = {'binspectra_path': binspectra_path, 
                 'fullbin_path': fullbin_path, 'plot_path': plot_path,
                 'bininfo_path': bininfo_path, 'fiberinfo_path': fiberinfo_path,
                 'targets_path': targets_path, 'ir_path': ir_path,
                 'raw_cube_path': raw_cube_path, 'gal_name': gal_name,
                 'crop_region': crop_region}
    things_to_plot.append(plot_info)

    # decide whether to continue with script or skip to plotting
    if os.path.isfile(binspectra_path):
        if input_params['skip_rerun']=='yes':
            print '\nSkipping re-run of {}, plotting only'.format(gal_name)
            continue
        elif input_params['skip_rerun']=='no':
            print '\nRunning {} again, will overwrite output'.format(gal_name)
        else:
            raise Exception("skip_rerun must be yes or no")
    else:
        print '\nRunning {}'.format(gal_name)

    # get bin layout...
    print "  binning..."
    gal_info = mpio.get_gal_info(targets_path,gal_name)
    ma_bin = np.pi/2 - np.deg2rad(gal_info['pa']) #theta=0 at +x (east), ccwise
    fiber_radius = const.mitchell_fiber_radius.value
    ifuset_all = ifu.read_raw_datacube(raw_cube_path, gal_info, gal_name,
                                       ir_path=ir_path)
    # crop wavelength range and remove fibers
    ifuset_all.crop(crop_region)
    badfibers = np.genfromtxt(bad_fibers_path,dtype=int)
    badfibers.sort()
    goodfibers = list(ifuset_all.spectrumset.ids)
    print "  ignoring fibers: {}".format(', '.join(map(str, badfibers)))
    for badfiber in badfibers:
        try: goodfibers.remove(badfiber)
        except: print "Duplicate bad fiber number: {}".format(badfiber)
    ifuset = ifuset_all.get_subset(goodfibers)
    # do all the bins
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
    grouped_ids, radial_bounds, angular_bounds, bin_bounds = binned
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
    bin_fluxes = np.zeros(number_bins)
    fiber_ids = ifuset.spectrumset.ids
    fiber_binnumbers = {f: const.unusedfiber_bin_id for f in fiber_ids}
    fiber_binnumbers.update({f: const.badfiber_bin_id for f in badfibers})
    #Loop over bins to get collapsed spectra, and record fiber and bin info
    for bin_iter, fibers in enumerate(grouped_ids):
        fiber_binnumbers.update({f: bin_ids[bin_iter] for f in fibers})
        subset = ifuset.get_subset(fibers)
        binned = subset.spectrumset.collapse(id='666') #dummy id
        binned_data["spectra"][bin_iter, :] = binned.spectra
        binned_data["bad_data"][bin_iter, :] = binned.metaspectra["bad_data"]
        binned_data["noise"][bin_iter, :] = binned.metaspectra["noise"]
        binned_data["ir"][bin_iter, :] = binned.metaspectra["ir"]
        xs, ys = subset.coords.T
        fluxes = subset.spectrumset.compute_flux()
        #Final bin coords want +x=west (not east), so use -xs
        bin_coords[bin_iter,:] = binning.calc_bin_center(-xs,ys,fluxes,bin_type,
                                  pa=gal_info['pa'],rmin=np.min(radial_bounds))
        bin_fluxes[bin_iter] = np.average(fluxes)
    spec_unit = ifuset.spectrumset.spec_unit
    wave_unit = ifuset.spectrumset.wave_unit
    binned_comments = ifuset.spectrumset.comments.copy()
    binned_comments["binning"] = "spectra have been spatially binned"
    binned_specset = spec.SpectrumSet(spectra_unit=spec_unit,
                                      wavelength_unit=wave_unit,
                                      comments=binned_comments,
                                      name=bin_type,
                                      **binned_data)
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
    # save binned spectrum
    binned_specset.write_to_fits(binspectra_path)
    # save fiber number vs bin number, sorted
    dt = {'names':['fiberid','binid'],'formats':[int,int]}
    fiberinfo = np.zeros(len(fiber_binnumbers),dtype=dt)
    fiberinfo['fiberid'] = fiber_binnumbers.keys()
    fiberinfo['binid'] = fiber_binnumbers.values()
    isort = np.argsort(fiberinfo['fiberid'])
    comments = ['{} is for unused fibers'.format(const.unusedfiber_bin_id),
                '{} is for bad fibers'.format(const.badfiber_bin_id)]
    mpio.save_textfile(fiberinfo_path,fiberinfo[isort],{},comments,fmt='%1i')
    # save bin number vs number of fibers, bin center coords, and bin boundaries
    metadata = {'coord unit': ifuset.coords_unit,
                'ifu file': os.path.basename(raw_cube_path),
                'ifu file date': time.ctime(os.path.getmtime(raw_cube_path)),
                'ir file': os.path.basename(ir_path),
                'ir file date': time.ctime(os.path.getmtime(ir_path)),
                'threshold ar': aspect_ratio,
                'threshold s2n': s2n_threshold,
                'r best fullbin': fullbin_radius,
                'bin type': bin_type}
    metadata.update({'gal {}'.format(k): v for k,v in gal_info.iteritems()})
    binning.write_bininfo(bininfo_path,bin_ids,grouped_ids,bin_fluxes,
                          bin_coords,bin_bounds,**metadata)
    # do the full galaxy bins
    fullids = [0,-1,-2]
    greatfibers = [f for f in goodfibers
                   if not fiber_binnumbers[f]==const.unusedfiber_bin_id]
    bestfibers = [] # this is where I will make a symmetrical one
    for fiber in greatfibers:
        x,y = ifuset.get_subset(fiber).coords[0]
        if np.sqrt(x**2 + y**2) < fullbin_radius:
            bestfibers.append(fiber)
    binned_comments["binning"] = ("bin 0 contains all good fibers, "
                                  "bin -1 contains all binned fibers, "
                                  "bin -2 contains all binned fibers within "
                                  "radius {}.".format(fullbin_radius))
    fullbin_data = {}
    fullbin_shape = (len(fullids),ifuset.spectrumset.num_samples)
    fullbin_data['wavelengths'] = ifuset.spectrumset.waves
    fullbin_data['spectra_ids'] = fullids
    for key in ['spectra','bad_data','noise','ir']:
        fullbin_data[key] = np.zeros(fullbin_shape)
    for i,f in enumerate([goodfibers,greatfibers,bestfibers]):
        full_galaxy = ifuset.get_subset(f).spectrumset.collapse(id=fullids[i])
        fullbin_data['spectra'][i,:] = full_galaxy.spectra
        fullbin_data['bad_data'][i,:] = full_galaxy.metaspectra['bad_data']
        fullbin_data['noise'][i,:] = full_galaxy.metaspectra['noise']
        fullbin_data['ir'][i,:] = full_galaxy.metaspectra['ir']
    full_galaxy = spec.SpectrumSet(spectra_unit=spec_unit,
                                   wavelength_unit=wave_unit,
                                   comments=binned_comments,
                                   name="fullgalaxybins",
                                   **fullbin_data)
    full_galaxy.write_to_fits(fullbin_path)


for plot_info in things_to_plot:
    print '\n\n====================================='
    print 'Plotting {}'.format(plot_info['gal_name'])
    plot_s2_bin_mitchell(**plot_info)

print '\n\n====================================='
print '====================================='
