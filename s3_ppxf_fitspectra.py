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
import massivepy.io as mpio


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
    output_dir, gal_name = mpio.parse_paramfile_path(paramfile_path)
    input_params = utl.read_dict_file(paramfile_path)
    binned_cube_path = input_params['binned_cube_path']
    if not os.path.isfile(binned_cube_path):
        raise Exception("Data cube {} does not exist".format(binned_cube_path))
    elif os.path.splitext(binned_cube_path)[-1] != ".fits":
        raise Exception("Invalid cube {}, need .fits".format(binned_cube_path))
    templates_dir = input_params['templates_dir']
    if not os.path.isdir(templates_dir):
        raise Exception("Template dir {} not found".format(templates_dir))
    temps_to_use = input_params['use_templates']
    if temps_to_use == 'all':
        pass
    elif not os.path.isfile(temps_to_use):
        raise Exception("Template list {} not found".format(temps_to_use))
    compare_moments = input_params['compare_moments'] #only for plotting
    compare_bins = input_params['compare_bins'] #only for plotting

    run_name = input_params['run_name']
    run_type = input_params['run_type']
    if not (run_type=='full' or run_type=='bins'):
        raise Exception("Run type is 'full' or 'bins', not {}".format(run_type))
    bins_to_fit = input_params['bins_to_fit']
    ### should probably change these to not just "eval"
    if not bins_to_fit == 'all':
        bins_to_fit = eval(input_params['bins_to_fit'])
    fit_settings = {'add_deg': input_params['add_deg'],
                    'mul_deg': input_params['mul_deg'],
                    'num_moments': input_params['num_moments'],
                    'bias': input_params['bias']}
    fit_range = eval(input_params['fit_range'])
    gh_init = eval(input_params['gh_init'])
    mask = eval(input_params['mask'])
    num_trials = int(input_params['num_trials'])

    # construct output file names
    output_path_maker = lambda f,ext: os.path.join(output_dir,
                "{}-s3-{}-{}-{}.{}".format(gal_name,run_name,run_type,f,ext))
    output_paths_dict = {}
    output_paths_dict['temps'] = output_path_maker('temps','txt')
    output_paths_dict['reg'] = output_path_maker('reg','fits')
    output_paths_dict['mc'] = output_path_maker('mc','fits')

    # save relevant info for plotting to a dict
    plot_info = {'reg_output': output_paths_dict['reg'],
                 'mc_output': output_paths_dict['mc'],
                 'temps_output': output_paths_dict['temps'],
                 'plot_path': output_path_maker('plots','pdf'),
                 'binspectra_path': binned_cube_path,
                 'run_type': run_type, 'templates_dir': templates_dir}
    things_to_plot.append(plot_info)

    # decide whether to continue with script or skip to plotting
    # only checks for "regular" fits file, not whether params have changed
    if os.path.isfile(output_paths_dict['reg']):
        if input_params['skip_rerun']=='yes':
            print '\nSkipping re-run of {}, plotting only'.format(gal_name)
            continue
        elif input_params['skip_rerun']=='no':
            print '\nRunning {} again, will overwrite output'.format(gal_name)
        else:
            raise Exception("skip_rerun must be yes or no")

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
    driver.write_outputs(output_paths_dict)


for plot_info in things_to_plot:
    plot_path = plot_info['plot_path']

    reg_data, reg_headers = utl.fits_quickread(plot_info['reg_output'])
    moments, lsq, scaledlsq = reg_data[0]
    nbins = moments.shape[0]
    nmoments = moments.shape[1]
    moment_names = ['h{}'.format(m+1) for m in range(nmoments)]
    moment_names[0] = 'V'
    moment_names[1] = 'sigma'

    # still need to clean up comparison plotting
    #if not plot_info['compare_moments']=='none':
    #    do_comparison = True
    #else:
    #    do_comparison = False
    #if do_comparison:
    #    fid_data,fid_headers = utl.fits_quickread(plot_info['compare_moments'])
    #    fid_moments, fid_lsq, fid_scaledlsq = fid_data[0]
    #    if not nbins==fid_moments.shape[0]:
    #        print 'Current {}, fid {}'.format(nbins,fid_moments.shape[0])
    #        raise Exception('Comparing to something with different bins, oops!')

    pdf = PdfPages(plot_path)
    # moments plots, for case of fitting all bins
    if plot_info['run_type']=='bins':
        # assuming the binspectra path ends in spectra.fits, this is not ideal
        bininfo_path = plot_info['binspectra_path'][0:-12]+'bininfo.txt'
        bininfo = np.genfromtxt(bininfo_path,names=True)
        if not len(bininfo)==nbins:
            raise Exception("Bin mismatch {},{}".format(nbins,len(bininfo)))
        for i in range(nmoments):
            fig = plt.figure(figsize=(6,5))
            fig.suptitle('Moment vs radius ({})'.format(moment_names[i]))
            ax = fig.add_axes([0.15,0.1,0.8,0.8])
            ax.plot(bininfo['r'],moments[:,i],ls='',marker='o',
                    c='b',ms=5.0,alpha=0.8)
            # comparison plot ability needs updating
            #if do_comparison:
            #    ax.plot(bininfo['r'],fid_moments[:,i],ls='',
            #            marker='s',c='g',ms=5.0,alpha=0.8,
            #            label='(old run)')
            #    ax.legend()
            #Symmetrize y axis for all but v and sigma
            if not i in (0,1):
                ylim = max(np.abs(ax.get_ylim()))
                ax.set_ylim(ymin=-ylim,ymax=ylim)
            ax.set_xlabel('radius')
            ax.set_ylabel('h{}'.format(i+1))
            pdf.savefig(fig)
            plt.close(fig)            
    # template plots, for full galaxy case
    elif plot_info['run_type']=='full':
        template_info = np.genfromtxt(plot_info['temps_output'],unpack=True)
        templates, weights, fluxes, fluxweights = template_info
        templates = templates.astype(int)
        catalogfile = os.path.join(plot_info['templates_dir'],'catalog.txt')
        catalog = pd.read_csv(catalogfile,index_col='miles_id')
        spectype = np.array(catalog['spt'][templates],dtype='S1')
        # sort by spectype for pie chart
        ii = np.argsort(spectype)
        templates = templates[ii]
        weights = weights[ii]
        fluxweights = fluxweights[ii]
        spectype = spectype[ii]
        spt_long = catalog['spt'][templates]
        pielabels = ["{} ({})".format(s,t) for s,t in zip(spt_long,templates)]
        piecolors = [const.spectype_colors[s[0]] for s in spectype]
        # plot raw weights
        fig = plt.figure(figsize=(6,5))
        fig.suptitle('Templates (raw weights)')
        ax = fig.add_axes([0.17,0.05,0.7,0.7*1.2])
        patches, labels, txt = ax.pie(weights,labels=pielabels,
                                      colors=piecolors,labeldistance=1.1,
                                      autopct='%1.1f%%',wedgeprops={'lw':0.2})
        for label in labels: label.set_fontsize(7)
        pdf.savefig(fig)
        # plot flux-normalized weights
        fig = plt.figure(figsize=(6,5))
        fig.suptitle('Templates (flux-normalized weights)')
        ax = fig.add_axes([0.15,0.05,0.7,0.7*1.2])
        patches, labels, txt = ax.pie(fluxweights,labels=pielabels,
                                      colors=piecolors,labeldistance=1.1,
                                      autopct='%1.1f%%',wedgeprops={'lw':0.2})
        for label in labels: label.set_fontsize(7)
        pdf.savefig(fig)
    # spectra for each bin, same in both cases
    ### want to change this to all spectra on one axis
    ### want to have bin ids, not just indexes
    ### want to have wavelength, not just pixel number
    spec, noise, usepix, modelspec, mulpoly, addpoly = reg_data[2]
    for b in range(nbins):
        fig = plt.figure(figsize=(6,5))
        fig.suptitle('Spectrum for bin {} of {}'.format(b+1, nbins))
        ax = fig.add_axes([0.15,0.1,0.8,0.8])
        ax.plot(spec[b,:],c='k',label='spectrum')
        ax.plot(modelspec[b,:],c='r',label='model spectrum')
        # find regions to mask
        maskpix = np.where(usepix[b,:]==0)[0]
        ibreaks = np.where(np.diff(maskpix)!=1)[0]
        maskpix_starts = [maskpix[0]]
        maskpix_starts.extend(maskpix[ibreaks+1])
        maskpix_ends = list(maskpix[ibreaks])
        maskpix_ends.append(maskpix[-1])
        for startpix,endpix in zip(maskpix_starts,maskpix_ends):
            ax.axvspan(startpix,endpix,fc='k',ec='none',alpha=0.5,lw=0)
        ax.set_xlabel('pixel')
        ax.set_ylabel('flux')
        pdf.savefig(fig)
        plt.close(fig)

    pdf.close()
