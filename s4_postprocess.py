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
    v_choice = int(input_params['v_choice'])
    targets_path = '../all_my_input/target-positions.txt' # wanna get from bins
    bininfo_path = input_params['bininfo_path']
    binfit_path = input_params['binfit_path']
    fullfit_path = input_params['fullfit_path']
    check = mpio.pathcheck([bininfo_path,binfit_path,fullfit_path],
                           ['.txt','.fits','.fits'],gal_name)
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
    binfits = mpio.get_friendly_ppxf_output(binfit_path)
    fullfits = mpio.get_friendly_ppxf_output(fullfit_path)
    gal_info = mpio.get_gal_info(targets_path, gal_name) # wanna get from bins

    # create a container for all of my radial profiles
    n_rsteps = len(bindata['r'])
    dt = {'names': ['lastbin','toplot','rbin','rencl','sig',
                    'lam','lam_minv0','lam_maxv0'],
          'formats': ['i8'] + ['b'] + 6*['f8']}
    rprofiles = np.zeros(n_rsteps,dtype=dt)

    # populate the radius information
    ii = np.argsort(bindata['r'])
    rprofiles['lastbin'] = bindata['binid'][ii]
    rprofiles['rbin'] = bindata['r'][ii]
    rencl = bindata['rmax'][ii]
    jj = np.isnan(rencl)
    rencl[jj] = bindata['r'][ii][jj]
    rprofiles['rencl'] = rencl
    kk = np.nonzero(np.diff(rencl))
    rprofiles['toplot'] = False
    rprofiles['toplot'][kk] = True
    rprofiles['toplot'][-1] = True

    # here is the setup for the calculation of lambda
    r = bindata['r']
    vel = binfits['gh']['moment'][:,0]
    sigma = binfits['gh']['moment'][:,1]
    flux = bindata['flux']
    luminosity = flux*bindata['nfibers'] # skipping fiber_area since it cancels

    # get all the choices for V0 first
    v0_all = {'full{}'.format(binid): v for (binid,v) 
              in zip(fullfits['bins']['id'],fullfits['gh']['moment'][:,0])}
    v_fullbin_index = np.where(v_choice==fullfits['bins']['id'])[0][0]
    v0_all['fiducial'] = fullfits['gh']['moment'][v_fullbin_index,0]
    v0_all['wbinavg'] = np.average(binfits['gh']['moment'][:,0],
                                           weights=luminosity)
    v0_all['binavg'] = np.average(binfits['gh']['moment'][:,0])
    v0_all['wbinmed'] = utl.median(binfits['gh']['moment'][:,0],
                                              weights=luminosity)
    v0_all['binmed'] = utl.median(binfits['gh']['moment'][:,0])

    # then calculate lambda
    for label,v0 in zip(['lam','lam_minv0','lam_maxv0'],
                [v0_all['fiducial'],min(v0_all.values()),max(v0_all.values())]):
        lamR = post.calc_lambda(r,vel-v0,sigma,luminosity)
        rprofiles[label] = lamR['lam']
    
    # do sigma thing
    sig_fluxavg = post.calc_sigma(r,sigma,luminosity)
    rprofiles['sig'] = sig_fluxavg['sig']

    # obtain some useful single-number metadata
    plotprof = rprofiles[rprofiles['toplot'].astype(bool)]
    lam_re = np.interp(gal_info['re'],plotprof['rencl'],plotprof['lam'])
    lam_re2 = np.interp(0.5*gal_info['re'],plotprof['rencl'],plotprof['lam'])
    sig_re = np.interp(gal_info['re'],plotprof['rencl'],plotprof['sig'])
    sig_re2 = np.interp(0.5*gal_info['re'],plotprof['rencl'],plotprof['sig'])
    slowfast_cutoff = 0.31*np.sqrt(1-gal_info['ba'])
    metadata = {'lambda_re': lam_re,
                'lambda_halfre' : lam_re2,
                'sigma_re': sig_re,
                'sigma_halfre' : sig_re2,
                're': gal_info['re'],
                'ba': gal_info['ba'],
                'isslow': int(lam_re<slowfast_cutoff),
                'sf_cutoff': slowfast_cutoff}
    metadata.update({'v0_{}'.format(k):v for k,v in v0_all.iteritems()})

    # save the radial profiles
    comments = ['For now, lam uses the flux-weighted average V as V0',
                'lam is luminosity weighted except lam_fluxw',
                'sig is luminosity weighted average sigma within R']
    post.write_rprofiles(rprofiles_path,rprofiles,metadata,comments)


for plot_info in things_to_plot:
    print '\n\n====================================='
    print 'Plotting {}'.format(plot_info['gal_name'])
    plot_s4_postprocess(**plot_info)
