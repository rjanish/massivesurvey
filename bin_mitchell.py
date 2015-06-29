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
    bad_fibers_path = input_params['bad_fibers_path']
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
    ifuset_all = ifu.read_mitchell_datacube(proc_cube_path)
    badfibers = np.genfromtxt(bad_fibers_path,dtype=int)
    goodfibers = list(ifuset_all.spectrumset.ids)
    for badfiber in badfibers:
        goodfibers.remove(badfiber)
    ifuset = ifuset_all.get_subset(goodfibers)
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

    delta_lambda = (ifuset.spectrumset.spec_region[1] -
                    ifuset.spectrumset.spec_region[0])
    #Do the full galaxy bin here because it is so fast.
    full_galaxy = ifuset.spectrumset.collapse(
                                  weight_func=spec.SpectrumSet.compute_flux,
                                  norm_func=spec.SpectrumSet.compute_flux,
                                  norm_value=delta_lambda, id=0)
    full_galaxy.comments["Binning"] = ("this spectrum is the coadditon "
                                       "of all fibers in the galaxy")
    fullbindesc = "fullgalaxybin"
    full_galaxy.name = fullbindesc
    fullbin_output_filename = '{}-{}.fits'.format(ngc_name,fullbindesc)
    fullbin_output_path = os.path.join(destination_dir, fullbin_output_filename)
    full_galaxy.write_to_fits(fullbin_output_path)
    #Now do the bins
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
    fiber_ids = ifuset.spectrumset.ids
    fiber_binnumbers = {f: const.unusedfiber_bin_id for f in fiber_ids}
    fiber_binnumbers.update({f: const.badfiber_bin_id for f in badfibers})
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
    #Save bin number vs number of fibers, bin center coords, and bin boundaries
    bininfo_path = "{}_bininfo.txt".format(output_base)
    dt = {'names':['binid','nfibers','x','y','r','th',
                   'rmin','rmax','thmin','thmax'],
          'formats':2*['i4']+8*['f32']}
    bininfo = np.zeros(number_bins,dtype=dt)
    bininfo['binid'] = bin_ids
    bininfo['nfibers'] = [len(fibers) for fibers in grouped_ids]
    for i,coord in enumerate(['x','y','r','th']):
        bininfo[coord] = bin_coords[:,i]
    for i,bound in enumerate(['rmin','rmax','thmin','thmax']):
        bininfo[bound] = bin_bounds[i,:]
    np.savetxt(bininfo_path,bininfo,delimiter='\t',fmt=2*['%1i']+8*['%9.5f'],
               header=' '.join(dt['names']))
    print 'You may ignore the weird underflow error, it is not important.'

for data_paths in things_to_plot:
    basepath = data_paths['fits'][:-5]
    plot_path = "{}.pdf".format(basepath)
    proc_cube_path = data_paths['proc_cube']
    gal_name = os.path.basename(basepath)[0:7]
    fullbin_path = os.path.join(os.path.dirname(basepath),
                                '{}-fullgalaxybin.fits'.format(gal_name))

    fiberinfo_path = "{}_fiberinfo.txt".format(basepath)
    fiberids, binids = np.genfromtxt(fiberinfo_path,dtype=int,unpack=True)
    bininfo_path = "{}_bininfo.txt".format(basepath)
    bininfo = np.genfromtxt(bininfo_path,names=True)
    bininfo['x'] *= -1  #east-west reflect

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

    specset = spec.read_datacube(data_paths['fits'])
    specset_full = spec.read_datacube(fullbin_path)

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
    bincolors[const.badfiber_bin_id] = 'k'
    bincolors[const.unusedfiber_bin_id] = '0.7'
    #Loop over fibers
    for fiber_id,bin_id in zip(fiberids,binids):
        ax.add_patch(patches.Circle(fiber_coords[fiber_id,:],fibersize,
                                    fc=bincolors[bin_id],ec='none',alpha=0.8))
    #Loop over bins
    for bin_iter,bin_id in enumerate(bininfo['binid']):
        bincolor = bincolors[int(bin_id)]
        #Draw star at bincenter
        ms = 7.0 + 0.5*bininfo['nfibers'][bin_iter]
        if ms > 20.0: ms = 20.0
        mew = 1.0 + 0.05*bininfo['nfibers'][bin_iter]
        if mew > 2.0: mew = 2.0
        #ax.plot(bininfo['x'][bin_iter],bininfo['y'][bin_iter],ls='',
        #        marker='*',mew=mew,ms=ms,mec='k',mfc=bincolor)
        ms = 8.0
        mew = 1.0
        ax.plot(bininfo['x'][bin_iter],bininfo['y'][bin_iter],ls='',
                marker='s',mew=mew,ms=ms,mec='k',mfc=bincolor)
        ax.text(bininfo['x'][bin_iter]-0.2,bininfo['y'][bin_iter]-0.1,
                str(int(bin_id)),fontsize=6,
                horizontalalignment='center',verticalalignment='center')
        #Draw bin outline
        if not np.isnan(bininfo['rmin'][bin_iter]):
            amax_xy = np.pi - bininfo['thmin'][bin_iter] #east-west reflect
            amin_xy = np.pi - bininfo['thmax'][bin_iter] #east-west reflect
            bin_poly = geo_utils.polar_box(bininfo['rmin'][bin_iter], 
                                           bininfo['rmax'][bin_iter],
                                           np.rad2deg(amin_xy),
                                           np.rad2deg(amax_xy))
            #Also do a transparent fill in bincolor to make sure bins match
            #If the storage of bin boundaries breaks, this will help notice
            ax.add_patch(descartes.PolygonPatch(bin_poly,fc=bincolor,
                                                ec='none',alpha=0.5))
            ax.add_patch(descartes.PolygonPatch(bin_poly,fc='none',lw=1.5))
    #Draw ma
    rmax = np.nanmax(bininfo['rmax'])
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
    fig = plt.figure(figsize=(6,nbins))
    ax = fig.add_axes([0.05,0.05,0.9,0.9])
    for ibin in range(nbins):
        spectrum = specset.spectra[ibin,:] 
        ax.plot(specset.waves,spectrum-spectrum[0]+specset.ids[ibin],c='k')
    fullspectrum = specset_full.spectra[0,:] 
    ax.plot(specset_full.waves,fullspectrum-fullspectrum[0],c='k') #id=0
    ax.set_xlabel('wavelength ({})'.format(specset.wave_unit))
    ax.set_ylabel('bin number')
    ax.autoscale(tight=True)
    ax.set_ylim(ymin=-1,ymax=nbins+2)
    pdf.savefig(fig)
    plt.close(fig)
        
    pdf.close()
