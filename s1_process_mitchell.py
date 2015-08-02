"""
Process the raw Mitchell datacubes into a format more accessible
for binning and fitting.

The changes are:
 - Coordinates converted from (RA, DEC) to projected Cartesian arcsec
 - Arc frames are fit by Gaussian line profiles and replaced with the
   resulting samples of fwhm(lambda) for each fiber

input:
  takes one command line argument, a path to the input parameter text file
  s1_process_mitchell_params_example.txt is an example
  can take multiple parameter files if you want to process multiple galaxies
  (give one param file per galaxy)

output:
  one processed datacube for each input raw datacube
  one pdf with diagnostic plots
"""


import argparse
import re
import os
import functools

import numpy as np
import pandas as pd
import shapely.geometry as geo
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import astropy.io.fits as fits

import utilities as utl
import massivepy.constants as const
import massivepy.spectralresolution as res
import massivepy.IFUspectrum as ifu
import massivepy.io as mpio
import massivepy.gausshermite as gh
from massivepy.plot_s1 import plot_s1_process_mitchell

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
    run_name = input_params['run_name']
    if not mpio.pathcheck([paramfile_path,raw_cube_path],
                          ['.txt','.fits'],
                          gal_name):
        print 'Something is wrong with the input paths for {}'.format(gal_name)
        print 'Skipping to next galaxy.'
        continue
    # construct output file names
    output_path_maker = lambda f, ext: os.path.join(output_dir,
                            "{}-s1-{}-{}.{}".format(gal_name, run_name, f, ext))
    plot_path = output_path_maker('fibermaps','pdf')
    ir_path = output_path_maker('ir','txt')
    # save relevant info for plotting to a dict
    plot_info = {'plot_path': plot_path,'ir_path': ir_path,
                 'raw_cube_path': raw_cube_path,
                 'targets_path': input_params['target_positions'],
                 'gal_name':gal_name}
    things_to_plot.append(plot_info)

    # decide whether to continue with script or skip to plotting
    if os.path.isfile(ir_path):
        if input_params['skip_rerun']=='yes':
            print '\nSkipping re-run of {}, plotting only'.format(gal_name)
            continue
        elif input_params['skip_rerun']=='no':
            print '\nRunning {} again, will overwrite output'.format(gal_name)
        else:
            raise Exception("skip_rerun must be yes or no")
    else:
        print '\nRunning {}'.format(gal_name)

    # start processing
    ifuset, arcs = ifu.read_raw_datacube(raw_cube_path,
                                         input_params['target_positions'],
                                         gal_name,
                                         return_arcs=True)
    gal_center_ra = ifuset.coord_comments['galaxy center RA']
    gal_center_dec = ifuset.coord_comments['galaxy center DEC']
    gal_pa = ifuset.coord_comments['galaxy pa']
    print "\n{}".format(gal_name)
    print "  raw datacube: {}".format(raw_cube_path)
    print "        center: {}, {}".format(gal_center_ra,gal_center_dec)
    print "            pa: {}".format(gal_pa)
    redshift = ifuset.spectrumset.comments['redshift']
    inst_waves = ifuset.spectrumset.waves*(1+redshift)
    print "  fitting arc frames..."
    spec_res_samples = res.fit_arcset(inst_waves, arcs,
                                      const.mitchell_arc_centers,
                                      const.mitchell_nominal_spec_resolution)
    res.save_specres(ir_path, spec_res_samples, ifuset.spectrumset.comments)
    print "  saved ir to text file."

for plot_info in things_to_plot:
    print '\n\n====================================='
    print 'Plotting {}'.format(plot_info['gal_name'])
    plot_s1_process_mitchell(**plot_info)

print '\n\n====================================='
print '\n\n=====================================\n\n'
