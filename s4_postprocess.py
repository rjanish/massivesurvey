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
import massivepy.constants as const
import massivepy.io as mpio
import massivepy.postprocess as post
from massivepy.plot_s4 import plot_s4_postprocess

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
    lambda_path_med = output_path_maker('lambda-med','txt')
    lambda_path_jstyle = output_path_maker('lambda-jstyle','txt')
    rprofiles_path = output_path_maker('rprofiles','txt')
    # save relevant info for plotting to a dict
    plot_info = {'fits_path': fits_path,
                 'rprofiles_path': rprofiles_path,
                 'lambda_path': lambda_path,
                 'lambda_path_med': lambda_path_med,
                 'lambda_path_jstyle': lambda_path_jstyle,
                 'plot_path': plot_path,
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
    bininfo = np.genfromtxt(bininfo_path,names=True,skip_header=1)
    #bininfo = np.genfromtxt(bininfo_path,names=True,skip_header=12)
    fitdata = mpio.get_friendly_ppxf_output(ppxf_output_path)

    # here is a placeholder for the code to write the public fits file
    header = fits.Header()
    hdu = fits.PrimaryHDU(data=bininfo['flux'],header=header)
    fits.HDUList([hdu]).writeto(fits_path, clobber=True)


    # create a container for all of my radial profiles
    n_rsteps = len(bininfo['r'])
    dt = {'names': ['lastbin','rtoplot','rbin','rencl','lam','sig','lam_med','lam_old'],
          'formats': ['i8'] + ['b'] + 6*['f8']}
    rprofiles = np.zeros(n_rsteps,dtype=dt)
    ii = np.argsort(bininfo['r'])
    rprofiles['rbin'] = bininfo['r'][ii]
    outerbinr = bininfo['rmax'][ii]
    jj = np.isnan(outerbinr)
    outerbinr[jj] = bininfo['r'][ii][jj]
    rdiffs = np.diff(outerbinr)
    kk = np.nonzero(rdiffs)
    rprofiles['rtoplot'] = False
    rprofiles['rtoplot'][kk] = True
    rprofiles['rtoplot'][-1] = True

    # here is the calculation of lambda
    lamR = post.calc_lambda(bininfo['r'],fitdata['gh']['moment'][:,0],
                    fitdata['gh']['moment'][:,1],bininfo['flux'],Vnorm='median')
    np.savetxt(lambda_path,lamR,header=' '.join(lamR.dtype.names))
    rprofiles['lam_old'] = lamR['lam']
    # do again using median
    luminosities = bininfo['flux']*bininfo['nfibers']
    lamR_median = post.calc_lambda(bininfo['r'],fitdata['gh']['moment'][:,0],
                                  fitdata['gh']['moment'][:,1],luminosities,
                                  Vnorm='median')
    np.savetxt(lambda_path_med,lamR_median,header=' '.join(lamR.dtype.names))
    rprofiles['lam_med'] = lamR_median['lam']
    # do again with the correct weighting!
    luminosities = bininfo['flux']*bininfo['nfibers']
    lamR_new = post.calc_lambda(bininfo['r'],fitdata['gh']['moment'][:,0],
                                fitdata['gh']['moment'][:,1],luminosities)
    rprofiles['lam'] = lamR_new['lam']
    
    # do sigma thing
    sig_fluxavg = post.calc_sigma(bininfo['r'],fitdata['gh']['moment'][:,1],
                                  bininfo['flux'])
    rprofiles['sig'] = sig_fluxavg['sig']

    # save the radial profiles
    np.savetxt(rprofiles_path,rprofiles,header=' '.join(rprofiles.dtype.names))

for plot_info in things_to_plot:
    plot_s4_postprocess(**plot_info)
