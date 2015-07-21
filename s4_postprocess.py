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
#import shutil
#import re
import argparse
import functools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
#import pandas as pd
import astropy.io.fits as fits

import utilities as utl
#import massivepy.constants as const
#import massivepy.templates as temps
#import massivepy.spectrum as spec
#import massivepy.pPXFdriver as driveppxf
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

    # so, here I will process the input parameter file. it will need to
    # contain the ppxf fits output (which ones? just main?) obviously,
    # and the bininfo.txt file as Jenny requested, but what else?

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
    # save relevant info for plotting to a dict
    plot_info = {'fits_path': fits_path,
                 'lambda_path': lambda_path,
                 'plot_path': plot_path}
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

    # ingest some things
    bininfo = np.genfromtxt(bininfo_path,names=True,skip_header=1)
    fitdata = mpio.get_friendly_ppxf_output(ppxf_output_path)

    # now here's the part where I actually do stuff!
    # first I suppose I will dump the bininfo stuff into a fits as a
    # placeholder for the rest of the fits packaging stuff
    header = fits.Header()
    hdu = fits.PrimaryHDU(data=bininfo['flux'],header=header)
    fits.HDUList([hdu]).writeto(fits_path, clobber=True)

    # then I'm gonna calculate lambda.
    # lambda comes out as a function of R, and ingredients are R,V,sigma,flux

    def calc_lambda(R,V,sigma,flux):
        nbins = len(R)
        if not len(V)==nbins and len(sigma)==nbins and len(flux)==nbins:
            raise Exception('u broke it.')
        V -= np.average(V,weights=flux)
        # sort by radius
        ii = np.argsort(R)
        R = R[ii]
        V = V[ii]
        sigma = sigma[ii]
        flux = flux[ii]
        dt = {'names':['R','Vavg','RVavg','m2avg','Rm2avg','lam'],
              'formats':6*[np.float64]}
        output = np.zeros(nbins,dtype=dt)
        output['R'] = R
        for i in range(nbins):
            avg = functools.partial(np.average,weights=flux[:i+1])
            output['Vavg'][i] = avg(np.abs(V[:i+1]))
            output['RVavg'][i] = avg(R[:i+1]*np.abs(V[:i+1]))
            output['m2avg'][i] = avg(np.sqrt(V[:i+1]**2+sigma[:i+1]**2))
            output['Rm2avg'][i]=avg(R[:i+1]*np.sqrt(V[:i+1]**2+sigma[:i+1]**2))
            output['lam'][i] = output['RVavg'][i]/output['Rm2avg'][i]
        return output

    lamR = calc_lambda(bininfo['r'],fitdata['gh']['moment'][:,0],
                       fitdata['gh']['moment'][:,1],bininfo['flux'])
    np.savetxt(lambda_path,lamR,header=' '.join(lamR.dtype.names))

for plot_info in things_to_plot:
    plot_path = plot_info['plot_path']

    lamR = np.genfromtxt(plot_info['lambda_path'],names=True)
    prettynames = {'R': 'radius',
                   'Vavg': r'$\langle |V| \rangle$',
                   'RVavg': r'$\langle R |V| \rangle$',
                   'm2avg': r'$\langle \sqrt{V^2 + \sigma^2} \rangle$',
                   'Rm2avg': r'$\langle R \sqrt{V^2 + \sigma^2} \rangle$',
                   'lam': r'$\lambda_R$'}

    ### Plotting Begins! ###

    pdf = PdfPages(plot_path)

    for thing in lamR.dtype.names:
        fig = plt.figure(figsize=(6,5))
        fig.suptitle(prettynames[thing])
        ax = fig.add_axes([0.17,0.15,0.7,0.7])
        ax.plot(lamR['R'],lamR[thing])
        ax.set_xlabel('radius')
        ax.set_ylabel(prettynames[thing])
        pdf.savefig(fig)
        plt.close(fig)

    pdf.close()

