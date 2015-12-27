"""
Modified plotting and extra munging on s3 results, to test dither Vs
Run with s3 parameter file.
"""

import os
import argparse
#import shutil
#import functools

import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
import descartes

import massivepy.constants as const
import massivepy.spectrum as spec
import massivepy.io as mpio
import massivepy.plot_massive as mplt
import massivepy.binning as binning
import massivepy.templates as temps
import utilities as utl
from plotting.geo_utils import polar_box


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
    # parse input parameter file, generate other needed paths
    output_dir, gal_name = mpio.parse_paramfile_path(paramfile_path)
    input_params = utl.read_dict_file(paramfile_path)
    bininfo_path = input_params['bin_info_path']
    fiberinfo_path = bininfo_path[:-11] + 'fiberinfo.txt'
    run_name = input_params['run_name']
    main_output = os.path.join(output_dir,"{}-s3-{}-main.fits".format(gal_name,
                                                                      run_name))
    check = mpio.pathcheck([bininfo_path,fiberinfo_path,main_output],
                           ['.txt','.txt','.fits'],gal_name)
    if not check:
        print 'Something is wrong with the input paths for {}'.format(gal_name)
        print 'Skipping to next galaxy.'
        continue

    ########
    # Do some plotting
    ########

    plot_path = os.path.join(output_dir,"{}-sX-Vtest.pdf".format(gal_name))

    # get data from files
    fitdata = mpio.get_friendly_ppxf_output(main_output)
    nbins = fitdata['metadata']['nbins']
    bindata, binetc = binning.read_bininfo(bininfo_path)
    fiberdata = np.genfromtxt(fiberinfo_path,names=True,skip_header=1)
    dithers = np.zeros(nbins,dtype='int')
    dcolors = ['c','m','y']

    ### Plotting Begins! ###

    pdf = PdfPages(plot_path)

    # do V vs radius, color-coded by dither
    fig = plt.figure(figsize=(6,6))
    fig.suptitle('V vs radius (colored by dither)')
    ax = fig.add_axes([0.15,0.1,0.8,0.7])
    ax.set_xlabel('radius')
    ax.set_ylabel('V')
    for i in range(nbins):
        V = fitdata['gh']['moment'][i,0]
        Verr = fitdata['gh']['scalederr'][i,0]
        binid = fitdata['bins']['id'][i]
        if not bindata['binid'][i]==binid:
            print 'U BROKE ITTTT'
        r = bindata['r'][i]
        fiberid = fiberdata['fiberid'][list(fiberdata['binid']).index(binid)]
        dither = int(fiberid)/245
        dithers[i] = dither
        ax.errorbar(r,V,yerr=Verr,ls='',marker=None,ecolor='0.7')
        ax.plot(r,V,ls='',marker='o',mfc=dcolors[dither],ms=7.0,alpha=0.8)
        ax.text(r,V,str(binid),fontsize=5,
                horizontalalignment='center',verticalalignment='center')
    handles = []
    labels = []
    for i in range(max(dithers)+1):
        ii = dithers==i
        Vavg = np.average(fitdata['gh']['moment'][ii,0],
                          weights=1/fitdata['gh']['scalederr'][ii,0])
        ax.axhline(Vavg,c=dcolors[i])
        handles.append(patches.Patch(color=dcolors[i]))
        labels.append('D{}'.format(i+1))
    ax.legend(handles,labels,loc='lower center',bbox_to_anchor=(0.5,1),ncol=i+1)
    pdf.savefig(fig)
    plt.close(fig)            

    
    pdf.close()
    
