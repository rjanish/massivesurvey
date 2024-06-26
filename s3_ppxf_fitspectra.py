"""
This script accepts spectral datacubes and fits each spectrum therein
using pPXF.

input:
  takes one command line argument, a path to the input parameter text file
  ppxf_fitspectra_params_example.txt is an example
  can take multiple parameter files if you want to process multiple galaxies
  (give one param file per galaxy)

output:
  one or two fits files for each galaxy, depending on if mc runs are done
  "friendly" text file versions of important output for theorists
  one pdf of various diagnostic plots
"""

import os
import re
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

import utilities as utl
import massivepy.constants as const
import massivepy.templates as temps
import massivepy.spectrum as spec
import massivepy.pPXFdriver as driveppxf
import massivepy.io as mpio
from massivepy.plot_s3 import plot_s3_fullfit, plot_s3_binfit

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
    run_name = input_params['run_name']
    run_type = input_params['run_type']
    check = True
    binned_cube_path = input_params['binned_cube_path']
    templates_dir = input_params['templates_dir']
    if not mpio.pathcheck([binned_cube_path,templates_dir],
                          ['.fits',''],gal_name):
        check = False
    if run_type == 'full':
        bininfo_path = 'none'
    else:
        bininfo_path = input_params['bin_info_path']
        if not mpio.pathcheck([bininfo_path],['.txt'],gal_name):
            check = False
    if 'use_templates' in input_params:
        temps_to_use = input_params['use_templates']
        if not mpio.pathcheck([temps_to_use],['.txt'],gal_name):
            check = False
    else:
        temps_to_use = 'all'
    if 'compare_moments' in input_params: # only for plotting
        compare_labels = eval(input_params['compare_labels'])
        compare_moments = input_params['compare_moments']
        compare_bins = input_params['compare_bins']
        if not mpio.pathcheck([compare_moments, compare_bins],
                              ['.fits','.txt'],gal_name):
            check = False
    else:
        compare_labels, compare_moments, compare_bins = ['none','none','none']
    if not check:
        print 'Something is wrong with the input paths for {}'.format(gal_name)
        print 'Skipping to next galaxy.'
        continue
    if 'bins_to_fit' in input_params:
        bins_to_fit = eval(input_params['bins_to_fit'])
    else:
        bins_to_fit = 'all'
    if 'num_trials' in input_params:
        num_trials = int(input_params['num_trials'])
    else:
        num_trials = 0
    fit_settings = {'add_deg': input_params['add_deg'],
                    'mul_deg': input_params['mul_deg'],
                    'num_moments': input_params['num_moments'],
                    'bias': input_params['bias']}
    fit_range = eval(input_params['fit_range'])
    gh_init = eval(input_params['gh_init'])
    mask = eval(input_params['mask'])

    # construct output file names
    output_path_maker = lambda f,ext: os.path.join(output_dir,
                "{}-s3-{}-{}.{}".format(gal_name,run_name,f,ext))
    output_paths_dict = {}
    output_paths_dict['main'] = output_path_maker('main','fits')
    output_paths_dict['mc'] = output_path_maker('mc','fits')
    if run_type=='full':
        plotname = 'templates'
    elif run_type=='bins':
        plotname = 'moments'
    else:
        raise Exception("Run type is 'full' or 'bins', not {}".format(run_type))
    # save relevant info for plotting to a dict
    plot_info = {'main_output': output_paths_dict['main'],
                 'mc_output': output_paths_dict['mc'],
                 'temps_output': output_path_maker('temps','txt'),
                 'moments_output': output_path_maker('moments','txt'),
                 'mcmoments_output': output_path_maker('mcmoments','')[:-1],
                 'plot_path': output_path_maker(plotname,'pdf'),
                 'binspectra_path': binned_cube_path,
                 'run_type': run_type,
                 'templates_dir': templates_dir,
                 'bininfo_path': bininfo_path,
                 'gal_name': gal_name,
                 'compare_labels': compare_labels,
                 'compare_moments': compare_moments,
                 'compare_bins': compare_bins,
                 'fit_range': fit_range,
                 'mask':mask}
    things_to_plot.append(plot_info)

    # decide whether to continue with script or skip to plotting
    # only checks for "main" fits file, not whether params have changed
    if os.path.isfile(output_paths_dict['main']):
        if input_params['skip_rerun']=='yes':
            print '\nSkipping re-run of {}, plotting only'.format(gal_name)
            continue
        elif input_params['skip_rerun']=='no':
            print '\nRunning {} again, will overwrite output'.format(gal_name)
        else:
            raise Exception("skip_rerun must be yes or no")
    else:
        print '\nRunning {}'.format(gal_name)

    # process library
    print "loading library {}...".format(templates_dir)
    full_template_library = temps.read_miles_library(templates_dir)
    if temps_to_use == 'all':
        template_library = full_template_library
    else:
        temps_list = np.genfromtxt(temps_to_use,usecols=0)
        template_library = full_template_library.get_subset(temps_list)
    print ("loaded library of {} templates"
           "".format(template_library.spectrumset.num_spectra))
    # get data
    print "reading spectra to fit..."
    specset = spec.read_datacube(binned_cube_path)
    binned_cube_date = time.ctime(os.path.getmtime(binned_cube_path))
    masked = utl.in_union_of_intervals(specset.waves, mask)
    if mask:
        print "masking the regions:"
        for mask_interval in mask:
            print '  {}'.format(mask_interval)
    else:
        print 'no regions masked'
    for spec_iter in xrange(specset.num_spectra):
        specset.metaspectra["bad_data"][spec_iter, :] = (
            specset.metaspectra["bad_data"][spec_iter, :] | masked)
            # move this logic into pPXFdriver - regions to mask in the
            # fit are not necessarily bad data
    if bins_to_fit=='all':
        specset_to_fit = specset
    else:
        specset_to_fit = specset.get_subset(bins_to_fit)
    # do fits
    driver = driveppxf.pPXFDriver(specset=specset_to_fit,
                                  templib=template_library,
                                  fit_range=fit_range,
                                  initial_gh=gh_init,
                                  num_trials=num_trials,
                                  sourcefile=os.path.basename(binned_cube_path),
                                  sourcedate=binned_cube_date,
                                  **fit_settings)
    driver.run_fit()
    driver.write_outputs(output_paths_dict)


for plot_info in things_to_plot:
    print '\n\n====================================='
    print 'Plotting {}'.format(plot_info['gal_name'])
    run_type = plot_info.pop('run_type')
    if run_type=='full':
        mpio.friendly_temps(plot_info['main_output'],plot_info['temps_output'])
        del plot_info['compare_labels']
        del plot_info['compare_moments']
        del plot_info['compare_bins']
        del plot_info['mc_output']
        del plot_info['moments_output']
        del plot_info['mcmoments_output']
        del plot_info['bininfo_path']
        plot_s3_fullfit(**plot_info)
    elif run_type=='bins':
        mpio.friendly_moments(plot_info['main_output'],plot_info['mc_output'],
                              plot_info['moments_output'],
                              plot_info['mcmoments_output'])
        #del plot_info['templates_dir'] #this is silly, make it so these don't
        del plot_info['temps_output']  #save in the first place
        plot_s3_binfit(**plot_info)

print '\n\n====================================='
print '====================================='
