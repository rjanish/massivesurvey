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
                 'aspect_ratio': aspect_ratio, 's2n_threshold': s2n_threshold,
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
    ifuset_all = ifu.read_raw_datacube(raw_cube_path, targets_path, gal_name,
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
    gal_position, gal_pa, gal_re= mpio.get_gal_center_pa(targets_path, gal_name)
    ma_bin = np.pi/2 - np.deg2rad(gal_pa) #theta=0 at +x (=east), ccwise
    fiber_radius = const.mitchell_fiber_radius.value
    # do the full galaxy bin
    full_galaxy = ifuset.spectrumset.collapse(id=0)
    full_galaxy.comments["binning"] = ("this spectrum is the coadditon "
                                       "of all fibers in the galaxy")
    full_galaxy.name = "fullgalaxybin"
    full_galaxy.write_to_fits(fullbin_path)
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
                                        pa=gal_pa,rmin=np.min(radial_bounds))
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
    fiberheader = "Fiber id vs bin id. "
    fiberheader += "\n {} is for unused fibers".format(const.unusedfiber_bin_id)
    fiberheader += "\n {} is for bad fibers".format(const.badfiber_bin_id)
    fiberinfo = np.array([np.array(fiber_binnumbers.keys()),
                          np.array(fiber_binnumbers.values())])
    isort = np.argsort(fiberinfo[0,:])
    np.savetxt(fiberinfo_path,fiberinfo[:,isort].T,fmt='%1i',delimiter='\t',
               header=fiberheader)
    # save bin number vs number of fibers, bin center coords, and bin boundaries
    dt = {'names':['binid','nfibers','flux','x','y','r','th',
                   'rmin','rmax','thmin','thmax'],
          'formats':2*['i4']+9*['f32']}
    fmt = 2*['%1i']+9*['%9.5f']
    bininfo = np.zeros(number_bins,dtype=dt)
    bininfo['binid'] = bin_ids
    bininfo['nfibers'] = [len(fibers) for fibers in grouped_ids]
    bininfo['flux'] = bin_fluxes
    for i,coord in enumerate(['x','y','r','th']):
        bininfo[coord] = bin_coords[:,i]
    # convert thetas from "binning" units (ccwise from +x=east)
    #  to "map" units (ccwise/towards -x/east from +y=north)
    #  by switching min, max and doing th_map = pi/2 - th_binning
    for i,bound in enumerate(['rmin','rmax','thmax','thmin']):
        bininfo[bound] = bin_bounds[i,:]
    bininfo['thmin'] = np.rad2deg(np.pi/2 - bininfo['thmin'])
    bininfo['thmax'] = np.rad2deg(np.pi/2 - bininfo['thmax'])
    binheader = 'Columns are as follows:'
    binheader += '\n' + ' '.join(dt['names'])
    binheader += '\nCoordinate definitions:'
    binheader += '\n x-direction is west, y-direction is north'
    binheader += '\n units are {}'.format(ifuset.coords_unit)
    binheader += '\n theta=0 is defined at +y (north)'
    binheader += '\n theta increases counterclockwise (towards east)'
    binheader += '\n theta is expressed in degrees'
    binheader += '\nCenter Ra/Dec are {}, {}'.format(gal_position[0],
                                                     gal_position[1])
    binheader += '\nPA (degrees, above theta definition) is {}'.format(gal_pa)
    binheader += '\nNote that x,y are bin centers in cartesian coordinates,'
    binheader += '\n while r,th are bin centers in polar coordinates,'
    binheader += '\n and they do not represent the same points!'
    ifufilename = os.path.basename(ifuset.spectrumset.comments['rawfile'])
    ifufiledate = ifuset.spectrumset.comments['rawdate']
    irfiledate = time.ctime(os.path.getctime(ir_path))
    binheader += "\nSource file: {}".format(ifufilename)
    binheader += "\n from {}".format(ifufiledate)
    binheader += "\n with ir file {}".format(ir_path)
    binheader += "\n from {}".format(irfiledate)
    binheader += "\nAspect ratio and s2n were set as:"
    binheader += "\n {}, {}".format(aspect_ratio, s2n_threshold)
    np.savetxt(bininfo_path,bininfo,delimiter='\t',fmt=fmt,header=binheader)
    print 'You may ignore the weird underflow error, it is not important.'

for plot_info in things_to_plot:
    print '\n\n====================================='
    print 'Plotting {}'.format(plot_info['gal_name'])
    plot_s2_bin_mitchell(**plot_info)

print '\n\n====================================='
print '====================================='
