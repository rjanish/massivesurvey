"""
This script does post-processing on the ppxf fit output (and other outputs).
The main task is to repackage the data into one fits file that is fit for
public consumption.
This script also calculates lambda.

input:
  takes one command line argument, a path to the input parameter text file
  ppxf_fitspectra_params_example.txt is an example
  can take multiple parameter files if you want to process multiple galaxies
  (give one param file per galaxy)

output:
  stuff
"""

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import astropy.io.fits as fits

import utilities as utl
import massivepy.io as mpio
import massivepy.lambdaR as lam

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

    bininfo_path = input_params['bininfo_path']
    if not os.path.isfile(bininfo_path):
        raise Exception("File {} does not exist".format(bininfo_path))
    ppxf_output_path = input_params['ppxf_output_path']
    if not os.path.isfile(ppxf_output_path):
        raise Exception("File {} does not exist".format(ppxf_output_path))
    run_name = input_params['run_name']
    # construct output file names
    output_path_maker = lambda f,ext: os.path.join(output_dir,
                "{}-s4-{}-{}.{}".format(gal_name,run_name,f,ext))
    fits_path = output_path_maker('underconstruction','fits')
    lambda_path = output_path_maker('lambda','txt')
    plot_path = output_path_maker('lambda','pdf')
    lambda_path_2 = output_path_maker('lambda-2','txt')
    lambda_path_3 = output_path_maker('lambda-3','txt')
    # save relevant info for plotting to a dict
    plot_info = {'fits_path': fits_path,
                 'lambda_path': lambda_path,
                 'plot_path': plot_path,
                 'lambda_path_2': lambda_path_2,
                 'lambda_path_3': lambda_path_3,
                 'gal_name': gal_name}
    things_to_plot.append(plot_info)

    # decide whether to continue with script or skip to plotting
    # only checks for "main" fits file, not whether params have changed
    if os.path.isfile(fits_path):
        if input_params['skip_rerun']=='yes':
            print '\nSkipping re-run of {}, plotting only'.format(gal_name)
            continue
        elif input_params['skip_rerun']=='no':
            print '\nRunning {} again, will overwrite output'.format(gal_name)
        else:
            raise Exception("skip_rerun must be yes or no")

    # ingest required data
    #bininfo = np.genfromtxt(bininfo_path,names=True,skip_header=1)
    bininfo = np.genfromtxt(bininfo_path,names=True,skip_header=12)
    fitdata = mpio.get_friendly_ppxf_output(ppxf_output_path)

    # here is a placeholder for the code to write the public fits file
    header = fits.Header()
    hdu = fits.PrimaryHDU(data=bininfo['flux'],header=header)
    fits.HDUList([hdu]).writeto(fits_path, clobber=True)

    # here is the calculation of lambda
    lamR = lam.calc_lambda(bininfo['r'],fitdata['gh']['moment'][:,0],
                       fitdata['gh']['moment'][:,1],bininfo['flux'])
    np.savetxt(lambda_path,lamR,header=' '.join(lamR.dtype.names))
    # for comparison and testing purposes, do it again but without
    # subtracting out the overall V
    lamR_no_offset = lam.calc_lambda(bininfo['r'],fitdata['gh']['moment'][:,0],
                                 fitdata['gh']['moment'][:,1],bininfo['flux'],
                                 Vnorm='no_offset')
    np.savetxt(lambda_path_2,lamR_no_offset,header=' '.join(lamR.dtype.names))
    # do again using median
    lamR_median = lam.calc_lambda(bininfo['r'],fitdata['gh']['moment'][:,0],
                                  fitdata['gh']['moment'][:,1],bininfo['flux'],
                                  Vnorm='median')
    np.savetxt(lambda_path_3,lamR_median,header=' '.join(lamR.dtype.names))

for plot_info in things_to_plot:
    lamR = np.genfromtxt(plot_info['lambda_path'],names=True)
    lamR_no_offset = np.genfromtxt(plot_info['lambda_path_2'],names=True)
    lamR_median = np.genfromtxt(plot_info['lambda_path_3'],names=True)
    labels = {'R': 'radius',
              'Vavg': r'$\langle |V| \rangle$',
              'RVavg': r'$\langle R |V| \rangle$',
              'm2avg': r'$\langle \sqrt{V^2 + \sigma^2} \rangle$',
              'Rm2avg': r'$\langle R \sqrt{V^2 + \sigma^2} \rangle$',
              'lam': r'$\lambda_R$',
              'V':r'$V$',
              'Vraw':r'$V_{\rm raw}$',
              'sigma':r'$\sigma$',
              'flux':'flux'}

    ### Plotting Begins! ###

    pdf = PdfPages(plot_info['plot_path'])

    # plot the actual lambda first
    fig = plt.figure(figsize=(6,5))
    fig.suptitle(labels['lam'])
    ax = fig.add_axes([0.17,0.15,0.7,0.7])
    ax.plot(lamR['R'],lamR['lam'],c='b',label='flux avg')
    ax.plot(lamR_median['R'],lamR_median['lam'],c='g',label='median')
    ax.plot(lamR_no_offset['R'],lamR_no_offset['lam'],c='r',label='raw V')
    ax.set_xlabel('radius')
    ax.set_ylabel(labels['lam'])
    if gal_name in lam.Jgals:
        print 'Comparing to numbers from Jenny'
        Jgal = lam.Jgals[gal_name]
        ax.plot(Jgal['Re'],Jgal['lRe'],ls='',marker='s',c='g',label='Jenny')
        ax.plot(0.25*Jgal['Re'],Jgal['l.25Re'],ls='',marker='s',c='g')
    ax.legend()
    pdf.savefig(fig)
    plt.close(fig)

    # plot the intermediate steps as subplots on one page
    fig = plt.figure(figsize=(6,6))
    fig.suptitle(r"Intermediate Stepss (x-axis matching $\lambda_R$)")
    for i, thing in enumerate(['Vavg','RVavg','m2avg','Rm2avg']):
        ax = fig.add_subplot(2,2,i+1)
        ax.plot(lamR['R'],lamR[thing],c='b')
        ax.plot(lamR_no_offset['R'],lamR_no_offset[thing],c='r')
        ax.plot(lamR_median['R'],lamR_median[thing],c='g')
        ax.set_title(labels[thing])
        ax.set_xticklabels([])
    pdf.savefig(fig)
    plt.close(fig)

    # plot some of the important raw numbers
    fig = plt.figure(figsize=(6,6))
    fig.suptitle(r"Ingredients (x-axis matching $\lambda_R$)")
    for i, thing in enumerate(['V','Vraw','sigma','flux']):
        ax = fig.add_subplot(2,2,i+1)
        ax.plot(lamR['R'],lamR[thing],c='b')
        ax.plot(lamR_no_offset['R'],lamR_no_offset[thing],c='r')
        ax.plot(lamR_median['R'],lamR_median[thing],c='g')
        ax.set_title(labels[thing])
        ax.set_xticklabels([])
    pdf.savefig(fig)
    plt.close(fig)

    pdf.close()
