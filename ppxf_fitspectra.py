"""
This is a temporary testing script!

This script accepts spectral datacubes and fits each spectrum therein
using pPXF.

input:
  takes one command line argument, a path to the input parameter text file
  ppxf_fitspectra_params_example.txt is an example
  can take multiple parameter files if you want to process multiple galaxies
  (give one param file per galaxy)

output:
  nothing at the moment
"""

import os
import re
import argparse

import numpy as np

import utilities as utl
import massivepy.constants as const
import massivepy.templates as temps
import massivepy.spectrum as spec
import massivepy.pPXFdriver as driveppxf


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
    binned_cube_path = input_params['binned_cube_path']
    if not os.path.isfile(binned_cube_path):
        raise ValueError("Invalid datacube path {}, "
                         "must be .fits file".format(binned_cube_path))
    fit_settings = {'add_deg': input_params['add_deg'],
                    'mul_deg': input_params['mul_deg'],
                    'num_moments': input_params['num_moments'],
                    'bias': input_params['bias']}
    #Below is a hacky way to allow convenient lists (or lists of lists)
    #Eventually all lines with eval() should be replaced with something better
    fit_range = eval(input_params['fit_range'])
    gh_init = eval(input_params['gh_init'])
    num_trials = int(input_params['num_trials'])
    run_name = str(input_params['run_name'])
    if not input_params['bins_to_fit']=='all':
        bins_to_fit = eval(input_params['bins_to_fit'])
    else:
        bins_to_fit = input_params['bins_to_fit']
    mask = eval(input_params['mask'])
    templates_dir = input_params['templates_dir']
    full_template_library = temps.read_miles_library(templates_dir)
    if input_params['template_list']=='all':
        template_library = full_template_library
    else:
        template_library = full_template_library.get_subset(
            const.fullMILES_1600fullgalaxy_optimized)
    destination_dir = input_params['destination_dir']
    if not os.path.isdir(destination_dir):
        raise ValueError("Invalid destination dir {}".format(destination_dir))
    # get data
    specset = spec.read_datacube(binned_cube_path)
    masked = utl.in_union_of_intervals(specset.waves, mask)
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
                                  **fit_settings)
    driver.run_fit()
    results = {"main_input":driver.main_input,
               "main_rawoutput":driver.main_rawoutput,
               "main_procoutput":driver.main_procoutput,
               "mc_input":driver.mc_input,
               "mc_rawoutput":driver.mc_rawoutput,
               "mc_procoutput":driver.mc_procoutput}
    for output_name, data in results.iteritems():
        output_path = os.path.join(destination_dir,
                                   "{}-{}.p".format(run_name, output_name))
        utl.save_pickle(data, output_path)