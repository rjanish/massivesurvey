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
import massivepy.binning as binning
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
    run_name = input_params['run_name']
    bininfo_path = input_params['bininfo_path']
    binfit_path = input_params['binfit_path']
    check = mpio.pathcheck([bininfo_path,binfit_path],['.txt','.fits'],gal_name)
    if not check:
        print 'Something is wrong with the input paths for {}'.format(gal_name)
        print 'Skipping to next galaxy.'
        continue
    # construct output file names
    output_path_maker = lambda f,ext: os.path.join(output_dir,
                "{}-s4-{}-{}.{}".format(gal_name,run_name,f,ext))
    # fits_path = output_path_maker('public','fits')
    plot_path = output_path_maker('lambda','pdf')
    rprofiles_path = output_path_maker('rprofiles','txt')
    # save relevant info for plotting to a dict
    plot_info = {'rprofiles_path': rprofiles_path,
                 'plot_path': plot_path,
                 'gal_name': gal_name}
    things_to_plot.append(plot_info)

    # decide whether to continue with script or skip to plotting
    # only checks for "main" fits file, not whether params have changed
    if os.path.isfile(rprofiles_path):
        if input_params['skip_rerun']=='yes':
            print '\nSkipping re-run of {}, plotting only'.format(gal_name)
            continue
        elif input_params['skip_rerun']=='no':
            print '\nRunning {} again, will overwrite output'.format(gal_name)
        else:
            raise Exception("skip_rerun must be yes or no")
    else:
        print '\nRunning {}'.format(gal_name)

    # ingest required data
    bindata, binetc = binning.read_bininfo(bininfo_path)
    #bindata = np.genfromtxt(bininfo_path,names=True,skip_header=12)
    fitdata = mpio.get_friendly_ppxf_output(binfit_path)

    # create a container for all of my radial profiles
    n_rsteps = len(bindata['r'])
    dt = {'names': ['lastbin','toplot','rbin','rencl','sig',
                    'lam','lam_med','lam_fluxw'],
          'formats': ['i8'] + ['b'] + 6*['f8']}
    rprofiles = np.zeros(n_rsteps,dtype=dt)

    # populate the radius information
    ii = np.argsort(bindata['r'])
    rprofiles['rbin'] = bindata['r'][ii]
    rencl = bindata['rmax'][ii]
    jj = np.isnan(rencl)
    rencl[jj] = bindata['r'][ii][jj]
    rprofiles['rencl'] = rencl
    kk = np.nonzero(np.diff(rencl))
    rprofiles['toplot'] = False
    rprofiles['toplot'][kk] = True
    rprofiles['toplot'][-1] = True

    # here is the calculation of lambda
    r = bindata['r']
    vel = fitdata['gh']['moment'][:,0]
    sigma = fitdata['gh']['moment'][:,1]
    flux = bindata['flux']
    luminosity = flux*bindata['nfibers'] # skipping fiber_area since it cancels

    lamR = post.calc_lambda(r,vel,sigma,luminosity)
    rprofiles['lam'] = lamR['lam']
    lamR = post.calc_lambda(r,vel,sigma,luminosity,Vnorm='median')
    rprofiles['lam_med'] = lamR['lam']
    lamR = post.calc_lambda(r,vel,sigma,flux)
    rprofiles['lam_fluxw'] = lamR['lam']
    
    # do sigma thing
    sig_fluxavg = post.calc_sigma(r,sigma,flux)
    rprofiles['sig'] = sig_fluxavg['sig']

    # save the radial profiles
    metadata = {'fullbin_index': -1,'test_thing': 15.6}
    comments = ['For now, lam uses the flux-weighted average V as V0',
                'lam is luminosity weighted except lam_fluxw',
                'sig is flux-weighted average sigma within R']
    post.save_rprofiles(rprofiles_path,rprofiles,metadata,comments)

for plot_info in things_to_plot:
    print '\n\n====================================='
    print 'Plotting {}'.format(plot_info['gal_name'])
    plot_s4_postprocess(**plot_info)
