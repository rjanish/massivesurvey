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
import massivepy.io as mpio
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
    output_dir, gal_name = mpio.parse_paramfile_path(paramfile_path)
    input_params = utl.read_dict_file(paramfile_path)
    proc_cube_path = input_params['proc_mitchell_cube']
    if not os.path.isfile(proc_cube_path):
        raise Exception("Data cube {} does not exist".format(proc_cube_path))
    elif os.path.splitext(proc_cube_path)[-1] != ".fits":
        raise Exception("Invalid cube {}, must be .fits".format(proc_cube_path))
    bad_fibers_path = input_params['bad_fibers_path']
    if not os.path.isfile(bad_fibers_path):
        raise Exception("File {} does not exist".format(bad_fibers_path))
    target_positions = pd.read_csv(input_params['target_positions_path'],
                                   comment='#', sep="[ \t]+",
                                   engine='python')
    run_name = input_params['run_name']
    aspect_ratio = input_params['aspect_ratio']
    s2n_threshold = input_params['s2n_threshold']
    bin_type = input_params['bin_type']
    # construct output file names
    output_path_maker = lambda f,ext: os.path.join(output_dir,
                "{}-s2-{}-{}-{}.{}".format(gal_name,run_name,bin_type,f,ext))
    binspectra_path = output_path_maker('spectra','fits')
    fullbin_path = output_path_maker('fullgalaxybin','fits')
    bininfo_path = output_path_maker('bininfo','txt')
    fiberinfo_path = output_path_maker('fiberinfo','txt')
    plot_path = output_path_maker('maps','pdf')
    # save relevant info for plotting to a dict
    plot_info = {'binspectra_path': binspectra_path, 
                 'fullbin_path': fullbin_path, 'plot_path': plot_path,
                 'bininfo_path': bininfo_path, 'fiberinfo_path': fiberinfo_path,
                 'targets_path': input_params['target_positions_path'],
                 'proc_cube_path': proc_cube_path,'gal_name': gal_name,
                 'aspect_ratio': aspect_ratio, 's2n_threshold': s2n_threshold}
    things_to_plot.append(plot_info)

    # decide whether to continue with script or skip to plotting
    if os.path.isfile(binspectra_path):
        if input_params['skip_rerun']=='yes':
            print '\nSkipping re-run of {}, plotting only'.format(gal_name)
            continue
        elif input_params['skip_rerun']=='no':
            print '\nRunning {} again, will overwrite output'.format(gal_name)
        else:
            raise Exception("skip_rerun must be yes or no")

    # get bin layout
    print "  binning..."
    ifuset_all = ifu.read_mitchell_datacube(proc_cube_path)
    badfibers = np.genfromtxt(bad_fibers_path,dtype=int)
    goodfibers = list(ifuset_all.spectrumset.ids)
    for badfiber in badfibers:
        goodfibers.remove(badfiber)
    ifuset = ifuset_all.get_subset(goodfibers)
    gal_position = target_positions[target_positions.Name == gal_name]
    gal_pa = gal_position.PA_best.iat[0]
    ma_bin = np.pi/2 - np.deg2rad(gal_pa) #theta=0 at +x (=east), ccwise
    fiber_radius = const.mitchell_fiber_radius.value
    #Do the full galaxy bin here because it is so fast.
    full_galaxy = ifuset.spectrumset.collapse(id=0)
    full_galaxy.comments["Binning"] = ("this spectrum is the coadditon "
                                       "of all fibers in the galaxy")
    full_galaxy.name = "fullgalaxybin"
    full_galaxy.write_to_fits(fullbin_path)
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
    bin_fluxes = np.zeros(number_bins)
    fiber_ids = ifuset.spectrumset.ids
    fiber_binnumbers = {f: const.unusedfiber_bin_id for f in fiber_ids}
    fiber_binnumbers.update({f: const.badfiber_bin_id for f in badfibers})
    #Loop over bins to get collapsed spectra, and record fiber and bin info
    for bin_iter, fibers in enumerate(grouped_ids):
        fiber_binnumbers.update({f: bin_ids[bin_iter] for f in fibers})
        subset = ifuset.get_subset(fibers)
        binned = subset.spectrumset.collapse(id='666') #dummy id
        binned_data["spectra"][bin_iter, :] = binned.spectra
        binned_data["bad_data"][bin_iter, :] = binned.metaspectra["bad_data"]
        binned_data["noise"][bin_iter, :] = binned.metaspectra["noise"]
        binned_data["ir"][bin_iter, :] = binned.metaspectra["ir"]
        xs, ys = subset.coords.T
        fluxes = subset.spectrumset.compute_flux()
        #Final bin coords want +x=west (not east), so use -xs
        bin_coords[bin_iter,:] = binning.calc_bin_center(-xs,ys,fluxes,bin_type,
                                        pa=gal_pa,rmin=np.min(radial_bounds))
        bin_fluxes[bin_iter] = np.average(fluxes)
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
    # save binned spectrum
    binned_specset.write_to_fits(binspectra_path)
    # save fiber number vs bin number, sorted
    fiberinfo_header = "Fiber id vs bin id. "
    fiberinfo = np.array([np.array(fiber_binnumbers.keys()),
                          np.array(fiber_binnumbers.values())])
    isort = np.argsort(fiberinfo[0,:])
    np.savetxt(fiberinfo_path,fiberinfo[:,isort].T,fmt='%1i',delimiter='\t',
               header=fiberinfo_header)
    # save bin number vs number of fibers, bin center coords, and bin boundaries
    dt = {'names':['binid','nfibers','flux','x','y','r','th',
                   'rmin','rmax','thmin','thmax'],
          'formats':2*['i4']+9*['f32']}
    fmt = 2*['%1i']+9*['%9.5f']
    bininfo = np.zeros(number_bins,dtype=dt)
    bininfo['binid'] = bin_ids
    bininfo['nfibers'] = [len(fibers) for fibers in grouped_ids]
    bininfo['flux'] = bin_fluxes
    for i,coord in enumerate(['x','y','r','th']):
        bininfo[coord] = bin_coords[:,i]
    # convert thetas from "binning" units (ccwise from +x=east)
    #  to "map" units (ccwise/towards -x/east from +y=north)
    #  by switching min, max and doing th_map = pi/2 - th_binning
    for i,bound in enumerate(['rmin','rmax','thmax','thmin']):
        bininfo[bound] = bin_bounds[i,:]
    bininfo['thmin'] = np.pi/2 - bininfo['thmin']
    bininfo['thmax'] = np.pi/2 - bininfo['thmax']
    binheader = 'Coordinate definitions:'
    binheader += '\n x-direction is west, y-direction is north'
    binheader += '\n units are {}'.format(ifuset.coord_comments['coordunit'])
    binheader += '\n theta=0 is defined at +y (north)'
    binheader += '\n theta increases counterclockwise (towards east)'
    binheader += '\n theta is expressed in radians'
    binheader += '\nCenter Ra/Dec are {}, {}'.format(gal_position.Ra.iat[0],
                                                     gal_position.Dec.iat[0])
    binheader += '\nPA (degrees, above theta definition) is {}'.format(gal_pa)
    binheader += '\nNote that x,y are bin centers in cartesian coordinates,'
    binheader += '\n while r,th are bin centers in polar coordinates,'
    binheader += '\n and they do not represent the same points!'
    binheader += '\nColumns are as follows:'
    binheader += '\n' + ' '.join(dt['names'])
    np.savetxt(bininfo_path,bininfo,delimiter='\t',fmt=fmt,header=binheader)
    print 'You may ignore the weird underflow error, it is not important.'

for plot_info in things_to_plot:
    plot_path = plot_info['plot_path']
    fiberids, binids = np.genfromtxt(plot_info['fiberinfo_path'],
                                     dtype=int,unpack=True)
    bininfo = np.genfromtxt(plot_info['bininfo_path'],names=True,skip_header=12)
    nbins = len(bininfo)
    ma_line = open(plot_info['bininfo_path'],'r').readlines()[7]
    ma_theta = np.pi/2 + np.deg2rad(float(ma_line.strip().split()[-1]))

    ifuset = ifu.read_mitchell_datacube(plot_info['proc_cube_path'])
    fiber_coords = ifuset.coords.copy()
    coordunit = ifuset.coord_comments['coordunit']
    fibersize = const.mitchell_fiber_radius.value #Assuming units match!
    fiber_coords[:, 0] *= -1  # east-west reflect
    squaremax = np.amax(np.abs(ifuset.coords)) + fibersize

    specset = spec.read_datacube(plot_info['binspectra_path'])
    # note that the specset fluxes are all the same! :(
    #logbinfluxes = np.log10(specset.compute_flux())
    logbinfluxes = np.log10(bininfo['flux'])
    # use colorbar limits from fiber maps, for continuity
    logfiberfluxes = np.log10(ifuset.spectrumset.compute_flux())
    logfmax = max(logfiberfluxes)
    logfmin = min(logfiberfluxes)
    fluxunit = specset.integratedflux_unit
    fcmap = plt.cm.get_cmap('Reds')
    fgetcolor = lambda f: fcmap((f - logfmin)/(logfmax-logfmin))
    logbins2n = np.log10(specset.compute_mean_s2n())
    logfibers2n = np.log10(ifuset.spectrumset.compute_mean_s2n())
    logsmax = max(logfibers2n)
    logsmin = min(logfibers2n)
    logsmin_bins = min(logbins2n)
    scmap = plt.cm.get_cmap('Greens')
    sgetcolor = lambda s: scmap((s - logsmin)/(logsmax-logsmin))
    sgetcolor2 = lambda s: scmap((s - logsmin_bins)/(logsmax-logsmin_bins))

    specset_full = spec.read_datacube(plot_info['fullbin_path'])

    ### plotting begins ###
    pdf = PdfPages(plot_path)

    # plot bin map, bin flux, and bin s2n
    fig1 = plt.figure(figsize=(6,6))
    fig1.suptitle('Bin map (s2n {}, ar {})'.format(s2n_threshold, aspect_ratio))
    ax1 = fig1.add_axes([0.15,0.1,0.7,0.7])
    fig2 = plt.figure(figsize=(6,6))
    fig2.suptitle('Bin flux map')
    ax2 = fig2.add_axes([0.15,0.1,0.7,0.7])
    fig3 = plt.figure(figsize=(6,6))
    fig3.suptitle('Bin s2n map')
    ax3 = fig3.add_axes([0.15,0.1,0.7,0.7])
    fig4 = plt.figure(figsize=(6,6))
    fig4.suptitle('Bin s2n map (rescaled)')
    ax4 = fig4.add_axes([0.15,0.1,0.7,0.7])
    fig5 = plt.figure(figsize=(6,6))
    fig5.suptitle('Bin centers comparison (cartesian vs polar)')
    ax5 = fig5.add_axes([0.15,0.1,0.7,0.7])
    mycolors = ['b','g','c','m','r','y']
    bincolors = {}
    for binid in set(binids):
        bincolors[binid] = mycolors[binid % len(mycolors)]
    bincolors[const.badfiber_bin_id] = 'k'
    bincolors[const.unusedfiber_bin_id] = '0.7'
    # loop over fibers
    for fiber_id,bin_id in zip(fiberids,binids):
        ax1.add_patch(patches.Circle(fiber_coords[fiber_id,:],fibersize,
                                    fc=bincolors[bin_id],ec='none',alpha=0.8))
    # loop over bins
    for bin_iter,bin_id in enumerate(bininfo['binid']):
        bincolor = bincolors[int(bin_id)]
        # draw bin number at bin center
        xbin = -bininfo['r'][bin_iter]*np.sin(bininfo['th'][bin_iter])
        ybin = bininfo['r'][bin_iter]*np.cos(bininfo['th'][bin_iter])
        ax1.plot(xbin,ybin,ls='',marker='o',mew=1.0,ms=8.0,mec='k',mfc=bincolor)
        ax1.text(xbin-0.2,ybin-0.1,str(int(bin_id)),fontsize=5,
                horizontalalignment='center',verticalalignment='center')
        # draw bin center, both versions
        ax5.plot(bininfo['x'][bin_iter],bininfo['y'][bin_iter],
                 ls='',marker='s',mew=0,ms=5.0,mfc='r')
        ax5.plot(xbin,ybin,ls='',marker='o',mew=0,ms=5.0,mfc='k')
        # draw bin outline and flux/s2n maps
        if not np.isnan(bininfo['rmin'][bin_iter]):
            thmin = np.rad2deg(np.pi/2 + bininfo['thmin'][bin_iter])
            thmax = np.rad2deg(np.pi/2 + bininfo['thmax'][bin_iter])
            bin_poly = geo_utils.polar_box(bininfo['rmin'][bin_iter], 
                                           bininfo['rmax'][bin_iter],
                                           thmin,thmax)
            # also do a transparent fill in bincolor to make sure bins match
            # if the storage of bin boundaries breaks, this will help notice
            ax1.add_patch(descartes.PolygonPatch(bin_poly,fc=bincolor,
                                                 ec='none',alpha=0.5,zorder=-1))
            ax1.add_patch(descartes.PolygonPatch(bin_poly,fc='none',lw=1.5))
            ax2.add_patch(descartes.PolygonPatch(bin_poly,
                                fc=fgetcolor(logbinfluxes[bin_iter]),lw=1.5))
            ax3.add_patch(descartes.PolygonPatch(bin_poly,
                                fc=sgetcolor(logbins2n[bin_iter]),lw=1.5))
            ax4.add_patch(descartes.PolygonPatch(bin_poly,
                                fc=sgetcolor2(logbins2n[bin_iter]),lw=1.5))
            ax5.add_patch(descartes.PolygonPatch(bin_poly,fc='none',lw=1.5))
        else:
            ax2.add_patch(patches.Circle((bininfo['x'][bin_iter],
                    bininfo['y'][bin_iter]),fibersize,lw=0.25,
                    fc=fgetcolor(logbinfluxes[bin_iter])))
            ax3.add_patch(patches.Circle((bininfo['x'][bin_iter],
                    bininfo['y'][bin_iter]),fibersize,lw=0.25,
                    fc=sgetcolor(logbins2n[bin_iter])))
            ax4.add_patch(patches.Circle((bininfo['x'][bin_iter],
                    bininfo['y'][bin_iter]),fibersize,lw=0.25,
                    fc=sgetcolor2(logbins2n[bin_iter])))

    # draw ma
    rmax = np.nanmax(bininfo['rmax'])
    ax1.plot([-rmax*1.1*np.cos(ma_theta), rmax*1.1*np.cos(ma_theta)],
             [-rmax*1.1*np.sin(ma_theta), rmax*1.1*np.sin(ma_theta)],
             linewidth=1.5, color='r')
    ax1.axis([-squaremax,squaremax,-squaremax,squaremax])
    ax2.axis([-squaremax,squaremax,-squaremax,squaremax])
    ax3.axis([-squaremax,squaremax,-squaremax,squaremax])
    ax4.axis([-squaremax,squaremax,-squaremax,squaremax])
    ax5.axis([-squaremax,squaremax,-squaremax,squaremax])
    label_x = r'$\leftarrow$east ({}) west$\rightarrow$'.format(coordunit)
    label_y = r'$\leftarrow$south ({}) north$\rightarrow$'.format(coordunit)
    label_flux = r'flux (log 10 [{}])'.format(fluxunit)
    label_s2n = r's2n (log 10)'
    ax1.set_xlabel(label_x)
    ax1.set_ylabel(label_y)
    # do colorbars
    ax2C = fig2.add_axes([0.15,0.8,0.7,0.8])
    ax2C.set_visible(False)
    mappable_flux = plt.cm.ScalarMappable(cmap=fcmap)
    mappable_flux.set_array([logfmin,logfmax])
    fig2.colorbar(mappable_flux,orientation='horizontal',ax=ax2C,
                  label=label_flux)
    ax3C = fig3.add_axes([0.15,0.8,0.7,0.8])
    ax3C.set_visible(False)
    mappable_s2n = plt.cm.ScalarMappable(cmap=scmap)
    mappable_s2n.set_array([logsmin,logsmax])
    fig3.colorbar(mappable_s2n,orientation='horizontal',ax=ax3C,
                  label=label_s2n)
    ax4C = fig4.add_axes([0.15,0.8,0.7,0.8])
    ax4C.set_visible(False)
    mappable_s2n_bins = plt.cm.ScalarMappable(cmap=scmap)
    mappable_s2n_bins.set_array([logsmin_bins,logsmax])
    fig4.colorbar(mappable_s2n_bins,orientation='horizontal',ax=ax4C,
                  label=label_s2n)
    pdf.savefig(fig1)
    plt.close(fig1)
    pdf.savefig(fig2)
    plt.close(fig2)
    pdf.savefig(fig3)
    plt.close(fig3)
    pdf.savefig(fig4)
    plt.close(fig4)
    pdf.savefig(fig5)
    plt.close(fig5)


    # plot each spectrum, y-axis also represents bin number
    fig = plt.figure(figsize=(6,nbins))
    fig.suptitle('bin spectra by bin number')
    ax = fig.add_axes([0.05,0.05,0.9,0.9])
    for ibin in range(nbins):
        spectrum = specset.spectra[ibin,:] 
        ax.plot(specset.waves,specset.ids[ibin]-spectrum+spectrum[0],c='k')
    fullspectrum = specset_full.spectra[0,:] 
    ax.plot(specset_full.waves,-fullspectrum+fullspectrum[0],c='k') #id=0
    ax.set_xlabel('wavelength ({})'.format(specset.wave_unit))
    ax.set_ylabel('bin number')
    ax.autoscale(tight=True)
    ax.set_ylim(ymin=-2,ymax=nbins+1)
    ax.invert_yaxis()
    ax.tick_params(labeltop='on',top='on')
    pdf.savefig(fig)
    plt.close(fig)

    # plot ir for each bin
    fig = plt.figure(figsize=(6,6))
    fig.suptitle('ir for each bin')
    ax = fig.add_axes([0.15,0.1,0.7,0.7])
    bcmap = plt.cm.get_cmap('cool')
    for ibin in range(nbins):
        ax.plot(specset.waves,specset.metaspectra['ir'][ibin,:],
                c=bcmap(ibin/float(nbins)),alpha=0.7)
    axC = fig.add_axes([0.15,0.8,0.7,0.8])
    axC.set_visible(False)
    mappable_bins = plt.cm.ScalarMappable(cmap=bcmap)
    mappable_bins.set_array([0,nbins])
    fig.colorbar(mappable_bins,orientation='horizontal',ax=axC,
                 label='bin number')
    pdf.savefig(fig)
    plt.close(fig)


    # reproduce process_mitchell flux plots with bad fibers highlighted
    fig1 = plt.figure(figsize=(6,6))
    fig1.suptitle('flux map')
    ax1 = fig1.add_axes([0.15,0.1,0.7,0.7])
    fig2 = plt.figure(figsize=(6,6))
    fig2.suptitle('flux vs radius')
    ax2 = fig2.add_axes([0.15,0.1,0.7,0.7])
    fig3 = plt.figure(figsize=(6,6))
    fig3.suptitle('flux vs radius')
    ax3 = fig3.add_axes([0.15,0.1,0.7,0.7])
    #The first quantity we want is flux per fiber
    logfluxes = np.log10(ifuset.spectrumset.compute_flux())
    logfmax = max(logfluxes)
    logfmin = min(logfluxes)
    fluxunit = ifuset.spectrumset.integratedflux_unit
    fcmap = plt.cm.get_cmap('Reds')
    fgetcolor = lambda f: fcmap((f - logfmin)/(logfmax-logfmin))
    fibertobindict = {f:b for (f,b) in zip(fiberids,binids)}
    rcoords = np.sqrt(fiber_coords[:,0]**2 + fiber_coords[:,1]**2)
    for ifiber in range(len(logfluxes)):
        fiber_id = ifuset.spectrumset.ids[ifiber]
        bin_id = fibertobindict[fiber_id]
        if not bin_id==const.badfiber_bin_id:
            ax1.add_patch(patches.Circle(fiber_coords[ifiber,:],
                                         fibersize,lw=0.25,
                                         fc=fgetcolor(logfluxes[ifiber])))
            ax2.text(rcoords[ifiber],logfluxes[ifiber],str(fiber_id),fontsize=5,
                     horizontalalignment='center',verticalalignment='center')
            ax3.text(rcoords[ifiber],logfluxes[ifiber],str(fiber_id),fontsize=5,
                     horizontalalignment='center',verticalalignment='center',
                     alpha=0.3)
        else:
            ax3.text(rcoords[ifiber],logfluxes[ifiber],str(fiber_id),fontsize=5,
                     horizontalalignment='center',verticalalignment='center')
            ax3.plot(rcoords[ifiber],logfluxes[ifiber],ls='',marker='o',
                     mec='r',mfc='none',ms=10,lw=1.0)
        ax1.text(fiber_coords[ifiber,0],fiber_coords[ifiber,1],
                 str(fiber_id),fontsize=5,
                 horizontalalignment='center',verticalalignment='center')
    ax1.axis([-squaremax,squaremax,-squaremax,squaremax])
    ax2.axis([min(rcoords),max(rcoords),min(logfluxes),max(logfluxes)])
    ax3.axis([min(rcoords),max(rcoords),min(logfluxes),max(logfluxes)])
    label_x = r'$\leftarrow$east ({}) west$\rightarrow$'.format(coordunit)
    label_y = r'$\leftarrow$south ({}) north$\rightarrow$'.format(coordunit)
    label_r = r'radius ({})'.format(coordunit)
    label_flux = r'flux (log 10 [{}])'.format(fluxunit)
    ax1.set_xlabel(label_x)
    ax1.set_ylabel(label_y)
    ax2.set_xlabel(label_r)
    ax2.set_ylabel(label_flux)
    ax3.set_xlabel(label_r)
    ax3.set_ylabel(label_flux)
    pdf.savefig(fig1)
    pdf.savefig(fig2)
    pdf.savefig(fig3)
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)

        
    pdf.close()
