"""
Construct polar, S/N threshold binning of Mitchell IFU fibers.

input:
  takes one command line argument, a path to the input parameter text file
  bin_mitchell_params_example.txt is an example
  can take multiple parameter files if you want to process multiple galaxies
  (give one param file per galaxy)

output:
  One binned datacube per galaxy. (One file contains all bins.)
  A bunch of other stuff that will be cleaned up at some point
"""


import os
import re
import argparse
import functools

import pickle

import numpy as np
import pandas as pd
import shapely.geometry as geo
import descartes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages


import utilities as utl
import massivepy.constants as const
import massivepy.IFUspectrum as ifu
import massivepy.spectrum as spec
import massivepy.binning as binning
import plotting.geo_utils as geo_utils


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
    proc_cube_path = input_params['proc_mitchell_cube']
    if ((not os.path.isfile(proc_cube_path))
        or (os.path.splitext(proc_cube_path)[-1] != ".fits")):
        raise ValueError("Invalid raw datacube path {}, "
                         "must be .fits file".format(proc_cube_path))
    aspect_ratio = input_params['aspect_ratio']
    s2n_threshold = input_params['s2n_threshold']
    target_positions = pd.read_csv(input_params['target_positions_path'],
                                   comment='#', sep="[ \t]+",
                                   engine='python')
    destination_dir = input_params['destination_dir']
    if not os.path.isdir(destination_dir):
        raise ValueError("Invalid destination dir {}".format(destination_dir))
    bin_type = input_params['bin_type']

    #New system for galaxy name, just taking from param file!!
    gal_name = os.path.basename(paramfile_path)[0:7]
    output_paths = {}
    output_paths['fits'] = os.path.join(destination_dir,
                                    "{}-{}.fits".format(gal_name,bin_type))
    output_paths['proc_cube'] = proc_cube_path
    output_paths['target_positions_path']=input_params['target_positions_path']
    things_to_plot.append(output_paths)
    if os.path.isfile(output_paths['fits']):
        if input_params['skip_rerun']=='yes':
            print '\nSkipping re-run of {}, plotting only'.format(gal_name)
            continue
        elif input_params['skip_rerun']=='no':
            print '\nRunning {} again, will overwrite output'.format(gal_name)
        else:
            raise Exception("skip_rerun must be yes or no")

    # get bin layout
    ifuset = ifu.read_mitchell_datacube(proc_cube_path)
    ngc_match = re.search(const.re_ngc, proc_cube_path)
    if ngc_match is None:
        msg = "No galaxy name found for path {}".format(proc_cube_path)
        raise RuntimeError(msg)
    else:
        ngc_num = ngc_match.groups()[0]
    ngc_name = "NGC{}".format(ngc_num)
    print "\n{}".format(ngc_name)
    print "  binning..."
    gal_position = target_positions[target_positions.Name == ngc_name]
    gal_pa = gal_position.PA_best.iat[0]
    ma_bin = np.pi/2 - np.deg2rad(gal_pa)
    ma_xy = np.pi/2 + np.deg2rad(gal_pa)
    fiber_radius = const.mitchell_fiber_radius.value
    if bin_type=='unfolded':
        apf = functools.partial(binning.partition_quadparity,
                                major_axis=ma_bin, aspect_ratio=aspect_ratio)
    elif bin_type=='folded':
        apf = functools.partial(binning.partition_quadparity_folded,
                                major_axis=ma_bin, aspect_ratio=aspect_ratio)
    else:
        raise Exception('Bin type must be folded or unfolded, try again.')
    binning_func = functools.partial(binning.polar_threshold_binning,
                                     angle_partition_func=apf)
    binned = ifuset.s2n_fluxweighted_binning(get_bins=binning_func,
                                             threshold=s2n_threshold)
    grouped_ids, radial_bounds, angular_bounds, bin_bounds = binned
    # results
    number_bins = len(grouped_ids)
    bin_ids = np.arange(number_bins, dtype=int) + 1  # bin 0 is full galaxy
    binned_data_shape = (number_bins, ifuset.spectrumset.num_samples)
    binned_data = {"spectra":np.zeros(binned_data_shape),
                   "bad_data":np.zeros(binned_data_shape),
                   "noise":np.zeros(binned_data_shape),
                   "ir":np.zeros(binned_data_shape),
                   "spectra_ids":bin_ids, # TO DO: add radial sorting
                   "wavelengths":ifuset.spectrumset.waves}
    bin_coords = np.zeros((number_bins, 4))  # flux weighted x, y, r, theta
    delta_lambda = (ifuset.spectrumset.spec_region[1] -
                    ifuset.spectrumset.spec_region[0])
    fiber_ids = ifuset.spectrumset.ids
    fiber_binnumbers = {f: const.unusedfiber_bin_id for f in fiber_ids}
    #Loop over bins to get collapsed spectra, and record fiber and bin info
    for bin_iter, fibers in enumerate(grouped_ids):
        fiber_binnumbers.update({f: bin_ids[bin_iter] for f in fibers})
        subset = ifuset.get_subset(fibers)
        binned = subset.spectrumset.collapse(
                                 weight_func=spec.SpectrumSet.compute_flux,
                                 norm_func=spec.SpectrumSet.compute_flux,
                                 norm_value=delta_lambda,id='666') #dummy id
        binned_data["spectra"][bin_iter, :] = binned.spectra
        binned_data["bad_data"][bin_iter, :] = binned.metaspectra["bad_data"]
        binned_data["noise"][bin_iter, :] = binned.metaspectra["noise"]
        binned_data["ir"][bin_iter, :] = binned.metaspectra["ir"]
        xs, ys = subset.coords.T
        fluxes = subset.spectrumset.compute_flux()
        bin_coords[bin_iter,:] = binning.calc_bin_center(xs,ys,fluxes,bin_type,
                                        ma=ma_bin,rmin=np.min(radial_bounds))
    spec_unit = ifuset.spectrumset.spec_unit
    wave_unit = ifuset.spectrumset.wave_unit
    binned_comments = ifuset.spectrumset.comments.copy()
    binned_comments["binning"] = "spectra have been spatially binned"
    binned_specset = spec.SpectrumSet(spectra_unit=spec_unit,
                                      wavelength_unit=wave_unit,
                                      comments=binned_comments,
                                      name=bin_type,
                                      **binned_data)
    single_fiber_bins = [l for l in grouped_ids if len(l) == 1]
    flat_binned_fibers = [f for l in grouped_ids for f in l]
    unbinned_fibers = [f for f in fiber_ids if f not in flat_binned_fibers]
    # output
    print "  {} total number of bins".format(len(grouped_ids))
    print "  {} single-fiber bins".format(len(single_fiber_bins))
    print "  {} un-binned outer fibers".format(len(unbinned_fibers))
    print "  multi-fiber layout:"
    for iter, [(rin, rout), angles] in enumerate(zip(radial_bounds,
                                                     angular_bounds)):
        print ("   {:2d}: radius {:4.1f} to {:4.1f}, {} angular bins"
               "".format(iter + 1, rin, rout, len(angles)))
    output_base = os.path.join(destination_dir,
                               "{}-{}".format(ngc_name, bin_type))
    binned_data_path = "{}.fits".format(output_base)
    binned_specset.write_to_fits(binned_data_path)
    #Save fiber number vs bin number, sorted
    fiberinfo_path = "{}_fiberinfo.txt".format(output_base)
    fiberinfo_header = "Fiber id vs bin id. "
    fiberinfo = np.array([np.array(fiber_binnumbers.keys()),
                          np.array(fiber_binnumbers.values())])
    isort = np.argsort(fiberinfo[0,:])
    np.savetxt(fiberinfo_path,fiberinfo[:,isort].T,fmt='%1i',delimiter='\t',
               header=fiberinfo_header)
    #Save bin number vs bin center coords and bin boundaries
    bininfo_path = "{}_bininfo.txt".format(output_base)
    bininfo = np.zeros((number_bins,9))
    bininfo[:,0] = bin_ids
    bininfo[:,1:5] = bin_coords
    bininfo[:,-4:] = bin_bounds.T
    bininfo_header = "Bin id, x/y/r/th coords, rmin/rmax/thmin/thmax bounds"
    np.savetxt(bininfo_path,bininfo,delimiter='\t',fmt=['%1i']+8*['%9.5f'],
               header=bininfo_header)
    ###Here be pickle files, BEWARE
    binned_path = "{}_binfibers.p".format(output_base)
    utl.save_pickle(grouped_ids, binned_path)
    unbinned_path = "{}_unbinnedfibers.p".format(output_base)
    utl.save_pickle(unbinned_fibers, unbinned_path)
    rad_path = "{}_radialbounds.p".format(output_base)
    utl.save_pickle(radial_bounds, rad_path)
    coord_path = "{}_fluxcenters.p".format(output_base)
    utl.save_pickle(bin_coords, coord_path)
    for anulus_iter, angle_bounds in enumerate(angular_bounds):
        angle_path = "{}_angularbounds-{}.p".format(output_base, anulus_iter)
        utl.save_pickle(angle_bounds, angle_path)
    ###End of pickle file territory
    print 'You may ignore the weird underflow error, it is not important.'

for data_paths in things_to_plot:
    basepath = data_paths['fits'][:-5]
    plot_path = "{}.pdf".format(basepath)
    proc_cube_path = data_paths['proc_cube']
    gal_name = os.path.basename(basepath)[0:7]

    fiberinfo_path = "{}_fiberinfo.txt".format(basepath)
    fiberids, binids = np.genfromtxt(fiberinfo_path,dtype=int,unpack=True)
    bininfo_path = "{}_bininfo.txt".format(basepath)
    bininfo = np.genfromtxt(bininfo_path)
    bininfo[:, 1] *= -1  #east-west reflect

    specset = spec.read_datacube(data_paths['fits'])
    #print specset.__dict__.keys()
    nbins = len(bininfo)
    ifuset = ifu.read_mitchell_datacube(proc_cube_path)
    fiber_coords = ifuset.coords.copy()
    coordunit = ifuset.coord_comments['coordunit']
    fibersize = const.mitchell_fiber_radius.value #Assuming units match!
    fiber_coords[:, 0] *= -1  # east-west reflect
    squaremax = np.amax(np.abs(ifuset.coords)) + fibersize

    target_positions = pd.read_csv(data_paths['target_positions_path'],
                                   comment='#', sep="[ \t]+",
                                   engine='python')
    gal_position = target_positions[target_positions.Name == gal_name]
    gal_pa = gal_position.PA_best.iat[0]
    ma_bin = np.pi/2 - np.deg2rad(gal_pa)
    ma_xy = np.pi/2 + np.deg2rad(gal_pa)

    ###Plotting begins!!!
    pdf = PdfPages(plot_path)

    fig = plt.figure(figsize=(6,6))
    fig.suptitle('Bin map (add s2n and ar here)')
    ax = fig.add_axes([0.15,0.1,0.7,0.7])
    #Define colors for each bin
    mycolors = ['b','g','c','m','r','y']
    bincolors = {}
    for binid in set(binids):
        bincolors[binid] = mycolors[binid % len(mycolors)]
    bincolors[const.badfiber_bin_id] = 'w'
    bincolors[const.unusedfiber_bin_id] = '0.7'
    #Loop over fibers
    for fiberid,binid in zip(fiberids,binids):
        ax.add_patch(patches.Circle(fiber_coords[fiberid,:],fibersize,
                                    fc=bincolors[binid],ec='none',alpha=0.8))
    #Loop over bins
    for bin_iter,bin_id in enumerate(bininfo[:,0]):
        #Draw star at bincenter
        #This is gonna break, need to add it to bininfo text file
        nfibers = 3
        ms = 7.0 + 0.5*nfibers
        if ms > 20.0: ms = 20.0
        mew = 1.0 + 0.05*nfibers
        if mew > 2.0: mew = 2.0
        ax.plot(bininfo[bin_iter,1],bininfo[bin_iter,2],ls='',marker='*',
                mew=mew,ms=ms,mec='k',mfc=bincolors[int(bin_id)])
        #Draw bin outline
        if not np.isnan(bininfo[bin_iter,-1]):
            amax_xy = np.pi - bininfo[bin_iter,7]
            amin_xy = np.pi - bininfo[bin_iter,8]
            bin_poly = geo_utils.polar_box(bininfo[bin_iter,5], 
                                           bininfo[bin_iter,6],
                                           np.rad2deg(amin_xy),
                                           np.rad2deg(amax_xy))
            ax.add_patch(descartes.PolygonPatch(bin_poly, facecolor='none',
                                                linestyle='solid',
                                                linewidth=1.5))
    #Draw ma
    rmax = np.nanmax(bininfo[:,6])
    ax.plot([-rmax*1.1*np.cos(ma_xy), rmax*1.1*np.cos(ma_xy)],
            [-rmax*1.1*np.sin(ma_xy), rmax*1.1*np.sin(ma_xy)],
            linewidth=1.5, color='r')
    ax.axis([-squaremax,squaremax,-squaremax,squaremax])
    label_x = r'$\leftarrow$east ({}) west$\rightarrow$'.format(coordunit)
    label_y = r'$\leftarrow$south ({}) north$\rightarrow$'.format(coordunit)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    pdf.savefig(fig)
    plt.close(fig)

    #Ok, now gonna plot each spectrum. For now I guess just jam them 
    # all onto one page to make my life easier, pretty it up later.
    fig = plt.figure(figsize=(6,0.5*nbins))
    ax = fig.add_axes([0.05,0.05,0.9,0.9])
    for ibin in range(nbins):
        spectrum = specset.spectra[ibin,:] 
        ax.plot(specset.waves,spectrum-spectrum[0]+specset.ids[ibin],c='k')
    ax.set_xlabel('wavelength ({})'.format(specset.wave_unit))
    ax.set_ylabel('bin number')
    ax.autoscale(tight=True)
    pdf.savefig(fig)
    plt.close(fig)
        

    pdf.close()
