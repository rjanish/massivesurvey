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
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

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

# empty list for outputs to plot
things_to_plot = []

for paramfile_path in all_paramfile_paths:
    # parse input parameter file
    input_params = utl.read_dict_file(paramfile_path)
    binned_cube_path = input_params['binned_cube_path']
    if not os.path.isfile(binned_cube_path):
        raise ValueError("Invalid datacube path {}, "
                         "must be .fits file".format(binned_cube_path))
    run_name = str(input_params['run_name'])
    destination_dir = input_params['destination_dir']
    if not os.path.isdir(destination_dir):
        raise ValueError("Invalid destination dir {}".format(destination_dir))

    #This output paths are coded seperately in the driver write_outputs()
    # function, which is bad. Should consolidate code and determine names
    # here for everything.
    output_paths = {}
    output_paths['main'] = os.path.join(destination_dir,
                                        '{}-main.fits'.format(run_name))
    output_paths['mc'] = os.path.join(destination_dir,
                                        '{}-mc.fits'.format(run_name))
    output_paths['bin_fitsfile'] = binned_cube_path
    things_to_plot.append(output_paths)
    #Check only for the main fits file, the only one guaranteed to exist
    if os.path.isfile(output_paths['main']):
        if input_params['skip_rerun']=='yes':
            print '\nSkipping re-run of {}, plotting only'.format(run_name)
            continue
        elif input_params['skip_rerun']=='no':
            print '\nRunning {} again, will overwrite output'.format(run_name)
        else:
            raise Exception("skip_rerun must be yes or no")

    #If the run is not skipped to just plot, finish parsing params file
    fit_settings = {'add_deg': input_params['add_deg'],
                    'mul_deg': input_params['mul_deg'],
                    'num_moments': input_params['num_moments'],
                    'bias': input_params['bias']}
    #Below is a hacky way to allow convenient lists (or lists of lists)
    #Eventually all lines with eval() should be replaced with something better
    fit_range = eval(input_params['fit_range'])
    gh_init = eval(input_params['gh_init'])
    num_trials = int(input_params['num_trials'])
    if not input_params['bins_to_fit']=='all':
        bins_to_fit = eval(input_params['bins_to_fit'])
    else:
        bins_to_fit = input_params['bins_to_fit']
    mask = eval(input_params['mask'])
    print "processing template library..."
    templates_dir = input_params['templates_dir']
    print "loading library {}...".format(templates_dir)
    full_template_library = temps.read_miles_library(templates_dir)
    if input_params['template_list']=='all':
        template_library = full_template_library
    else:
        temps_to_use = eval(input_params["template_list"])
        template_library = full_template_library.get_subset(temps_to_use)
    print ("loaded library of {} templates"
           "".format(template_library.spectrumset.num_spectra))
    # get data
    print "reading spectra to fit..."
    specset = spec.read_datacube(binned_cube_path)
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
                                  **fit_settings)
    driver.run_fit()
    driver.write_outputs(destination_dir, run_name)


for data_paths in things_to_plot:
    plot_path = "{}.pdf".format(data_paths['main'][:-5])

    main_data, main_headers = utl.fits_quickread(data_paths['main'])
    moments, lsq, scaledlsq = main_data[0]
    nbins = moments.shape[0]
    nmoments = moments.shape[1]

    pdf = PdfPages(plot_path)
    #If this is a binned fit, we care about moment vs radius
    if nbins > 1:
        #This is terrible and will get better when we reorganize the output
        # of the binning code.
        bins_path = '{}_fluxcenters.p'.format(data_paths['bin_fitsfile'][:-5])
        bincenters_pickle = pickle.load(open(bins_path,'r'))
        for i in range(nmoments):
            fig = plt.figure(figsize=(6,5))
            fig.suptitle('Moment vs radius (h{})'.format(i+1))
            ax = fig.add_axes([0.15,0.1,0.8,0.8])
            ax.plot(bincenters_pickle[:,2],moments[:,i],ls='',marker='o')
            ax.set_xlabel('radius')
            ax.set_ylabel('h{}'.format(i+1))
            pdf.savefig(fig)
    #If there is only one bin, we care about templates
    else:
        fig = plt.figure(figsize=(6,6))
        fig.suptitle('Template stuff')
        ax = fig.add_axes([0.17,0.1,0.7,0.7])

        #Should have the fits file save only nonzero in the first place
        ii = np.nonzero(main_data[1][1,0,:])
        templates, weights, fluxes, fluxweights = main_data[1][:,0,ii]

        #Some hacky stuff to get template star information
        catalogfile = '../all_my_output/miles-processed/catalog.txt'
        catalog2 = pd.read_csv(catalogfile,index_col='miles_id')
        templates = templates[0].astype(int)
        spectype_colors = {'A':'aqua','B':'blue','G':'green','F':'lime',
                           'I':'indigo','M':'magenta','K':'crimson',
                           '-':'black','S':'orange','0':'gray','s':'tan',
                           'R':'yellow','H':'gold'}
        spectype = catalog2['spt'][templates]
        pielabels = ["{} ({})".format(s,t) for s,t in zip(spectype,templates)]
        piecolors = [spectype_colors[s[0]] for s in spectype]
        #print catalog.dtype.names
        patches, labels = ax.pie(weights[0],labels=pielabels,colors=piecolors,
                                 labeldistance=1.3,wedgeprops={'lw':0.2})
        for label in labels: label.set_fontsize(7)
        pdf.savefig(fig)

    pdf.close()
