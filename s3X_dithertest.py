"""
Modified plotting and extra munging on s3 results, to test dither Vs
Run with s3 parameter file.
"""

import os
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches

import massivepy.io as mpio
import massivepy.binning as binning
import utilities as utl


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

    output_path_maker = lambda f,ext: os.path.join(output_dir,
                            "{}-s3X-{}-{}.{}".format(gal_name,run_name,f,ext))
    plot_path = output_path_maker('vtest','pdf')
    shifts_path = output_path_maker('voffsets','txt')

    # get data from files
    fitdata = mpio.get_friendly_ppxf_output(main_output)
    nbins = fitdata['metadata']['nbins']
    bindata, binetc = binning.read_bininfo(bininfo_path)
    fiberdata = np.genfromtxt(fiberinfo_path,names=True,skip_header=1)
    dithers = np.zeros(nbins,dtype='int')
    dcolors = ['c','m','y','b','r','g','k']
    nperdither = 245

    ### Plotting Begins! ###

    pdf = PdfPages(plot_path)

    # do V vs radius, color-coded by dither
    fig = plt.figure(figsize=(6,6))
    fig.suptitle('V vs radius (colored by dither)')
    ax = fig.add_axes([0.15,0.1,0.8,0.7])
    ax.set_xlabel('radius')
    ax.set_ylabel('V')
    for i in range(nbins):
        if not bindata['nfibers'][i]==1:
            continue
        V = fitdata['gh']['moment'][i,0]
        Verr = fitdata['gh']['scalederr'][i,0]
        binid = fitdata['bins']['id'][i]
        r = bindata['r'][i]
        fiberid = fiberdata['fiberid'][list(fiberdata['binid']).index(binid)]
        dither = int(fiberid)/nperdither
        dithers[i] = dither
        ax.errorbar(r,V,yerr=Verr,ls='',marker=None,ecolor='0.7')
        ax.plot(r,V,ls='',marker='o',mfc=dcolors[dither],ms=7.0,alpha=0.8)
        ax.text(r,V,str(binid),fontsize=5,
                horizontalalignment='center',verticalalignment='center')
    handles = []
    labels = []
    ditherVs = []
    for i in range(max(dithers)+1):
        ii = dithers==i
        if sum(ii)==0:
            print 'All fibers from dither {} bad'.format(i+1)
            ditherVs.append(np.nan)
            continue
        Vavg = np.average(fitdata['gh']['moment'][ii,0],
                          weights=1/fitdata['gh']['scalederr'][ii,0])
        ax.axhline(Vavg,c=dcolors[i])
        ditherVs.append(Vavg)
        handles.append(patches.Patch(color=dcolors[i]))
        labels.append('D{}'.format(i+1))
    ax.legend(handles,labels,loc='lower center',bbox_to_anchor=(0.5,1),ncol=i+1)
    pdf.savefig(fig)
    plt.close(fig)            

    
    pdf.close()
    

    # save average Vs to a text file for machine readability
    header = "Average radial velocity for each dither, in km/s"
    header += "\n Assumes {} fibers per dither".format(nperdither)
    header += "\n Calculated from the following fit results:"
    header += "\n    {}".format(main_output)
    header += "\n    {}".format(time.ctime(os.path.getmtime(main_output)))
    np.savetxt(shifts_path,ditherVs,fmt='%6f',header=header)
