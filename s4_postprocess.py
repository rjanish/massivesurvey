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
    junk_bins = int(input_params['junk_bins'])
    targets_path = '../all_my_input/target-positions.txt' # wanna get from bins
    bininfo_path = input_params['bininfo_path']
    binfit_path = input_params['binfit_path']
    fullfit_path = input_params['fullfit_path']
    check = mpio.pathcheck([bininfo_path,binfit_path,fullfit_path],
                           ['.txt','.txt','.fits'],gal_name)
    if not check:
        print 'Something is wrong with the input paths for {}'.format(gal_name)
        print 'Skipping to next galaxy.'
        continue
    # construct output file names
    output_path_maker = lambda f,ext: os.path.join(output_dir,
                "{}-s4-{}-{}.{}".format(gal_name,run_name,f,ext))
    # fits_path = output_path_maker('public','fits')
    plot_path = output_path_maker('lambda','pdf')
    rdata_path = output_path_maker('rprofiles','txt')
    # save relevant info for plotting to a dict
    plot_info = {'rdata_path': rdata_path,
                 'binfit_path': binfit_path,
                 'plot_path': plot_path,
                 'gal_name': gal_name}
    things_to_plot.append(plot_info)

    # decide whether to continue with script or skip to plotting
    # only checks for "main" fits file, not whether params have changed
    if os.path.isfile(rdata_path):
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
    bindata, binmeta = binning.read_bininfo(bininfo_path)
    binmoments = np.genfromtxt(binfit_path,names=True,skip_header=1)
    fullfits = mpio.get_friendly_ppxf_output(fullfit_path)

    # erase junk bins like they never existed
    if not junk_bins==0:
        bindata = bindata[:-junk_bins]
        binmoments = binmoments[:-junk_bins]

    # get all the choices for V0 packaged up first
    luminosity = bindata['flux']*bindata['nfibers'] # skipping fiber_area

    v0_all = {'full{}'.format(binid): v for (binid,v) 
              in zip(fullfits['bins']['id'],fullfits['gh']['moment'][:,0])}
    v_fullbin_index = np.where(v_choice==fullfits['bins']['id'])[0][0]
    v0_all['fiducial'] = fullfits['gh']['moment'][v_fullbin_index,0]
    v0_all['wbinavg'] = np.average(binmoments['V'],weights=luminosity)
    v0_all['binavg'] = np.average(binmoments['V'])
    v0_all['wbinmed'] = utl.median(binmoments['V'],weights=luminosity)
    v0_all['binmed'] = utl.median(binmoments['V'])
    v0_min, v0_max = min(v0_all.values()), max(v0_all.values())


    # group annular bins and create the radial profiles
    bin_groups = post.group_bins(bindata)
    dt = {'names':['r','r_en','sig_loc','sig_en','sig_loc_err',
                   'lam_loc','lam_loc_vmin','lam_loc_vmax',
                   'lam_en','lam_en_vmin','lam_en_vmax'],
          'formats':11*['f8']}
    rdata = np.zeros(len(bin_groups),dtype=dt)
    group_en = []
    for i,group in enumerate(bin_groups):
        bd = bindata[group]
        bm = binmoments[group]
        if all(np.isnan(bd['rmax'])):
            rdata[i]['r_en'] = np.max(bd['r'])
        elif all(~np.isnan(bd['rmax'])):
            rdata[i]['r_en'] = bd['rmax'][0]
        else:
            raise Exception('Help, ur annulus is broke.')
        lum = bd['flux']*bd['nfibers']
        rdata[i]['r'] = np.average(bd['r'],weights=lum)
        rdata[i]['sig_loc'] = np.average(bm['sigma'],weights=lum)
        rdata[i]['sig_loc_err'] = np.std(bm['sigma'])
        for v0, label in zip([v0_all['fiducial'],v0_min,v0_max],
                             ['lam_loc','lam_loc_vmin','lam_loc_vmax']):
            rdata[i][label] = post.lam(bd['r'],bm['V']-v0,bm['sigma'],lum)
        # now do the cumulative things
        group_en.extend(group)
        bd = bindata[np.array(group_en)]
        bm = binmoments[np.array(group_en)]
        lum = bd['flux']*bd['nfibers']
        rdata[i]['sig_en'] = np.average(bm['sigma'],weights=lum)
        for v0, label in zip([v0_all['fiducial'],v0_min,v0_max],
                             ['lam_en','lam_en_vmin','lam_en_vmax']):
            rdata[i][label] = post.lam(bd['r'],bm['V']-v0,bm['sigma'],lum)


    # get the correlations and bootstrap error bars for h3/V
    voversigma = (binmoments['V']-v0_all['fiducial'])/binmoments['sigma']
    h3boots = post.bootstrap(voversigma,binmoments['h3'])
    sigma0 = np.average(binmoments['sigma'])
    sigmaoversigma = (binmoments['sigma']-sigma0)/sigma0
    h4boots = post.bootstrap(sigmaoversigma,binmoments['h4'])


    # obtain some useful single-number metadata
    lam_re = np.interp(binmeta['gal re'],rdata['r_en'],rdata['lam_en'])
    lam_re2 = np.interp(0.5*binmeta['gal re'],rdata['r_en'],rdata['lam_en'])
    sig_c = binmoments['sigma'][0]
    sig_re = np.interp(binmeta['gal re'],rdata['r_en'],rdata['sig_en'])
    sig_re2 = np.interp(0.5*binmeta['gal re'],rdata['r_en'],rdata['sig_en'])
    sig_err = np.max(rdata['sig_loc_err']/rdata['sig_loc'])
    slowfast_cutoff = 0.31*np.sqrt(1-binmeta['gal ba'])
    metadata = {'junk bins': junk_bins,
                'lambda re': lam_re,
                'lambda half re' : lam_re2,
                'sigma center': sig_c,
                'sigma re': sig_re,
                'sigma half re' : sig_re2,
                'sigma anisotropy' : sig_err,
                'gal re': binmeta['gal re'],
                'gal ba': binmeta['gal ba'],
                'is slow': int(lam_re<slowfast_cutoff),
                'slow/fast cutoff': slowfast_cutoff,
                'h3 slope': h3boots['slope'],
                'h3 slope err': h3boots['slope_err'],
                'h3 intercept': h3boots['intercept'],
                'h3 intercept err': h3boots['intercept_err'],
                'h4 slope': h4boots['slope'],
                'h4 slope err': h4boots['slope_err'],
                'h4 intercept': h4boots['intercept'],
                'h4 intercept err': h4boots['intercept_err'],
                'h3 average': np.average(binmoments['h3']),
                'h3 err': np.std(binmoments['h3']),
                'h4 average': np.average(binmoments['h4']),
                'h4 err': np.std(binmoments['h4']),
                'h5 average': np.average(binmoments['h5']),
                'h5 err': np.std(binmoments['h5']),
                'h6 average': np.average(binmoments['h6']),
                'h6 err': np.std(binmoments['h6'])}
    metadata.update({'v0_{}'.format(k):v for k,v in v0_all.iteritems()})

    # save the radial profiles
    comments = [('Radial profiles have both local ("loc") and '
                 'cumulative/enclosed ("en") versions'),
                ('Lambda profiles are also calculated with min and '
                 'max V0 to verify that no major differences'),
                ('Included in metadata are (unweighted) averages '
                 'and standard deviations of h3-h6'),
                ('H3 and h4 slopes and intercepts are from bootstrapped '
                 'linear fits of h3/V and h4/sigma correlations'),
                ('Sigma anisotropy is the maximum relative deviation '
                 'of sigma within an annulus')]
    mpio.save_textfile(rdata_path,rdata,metadata,comments)


for plot_info in things_to_plot:
    print '\n\n====================================='
    print 'Plotting {}'.format(plot_info['gal_name'])
    plot_s4_postprocess(**plot_info)
