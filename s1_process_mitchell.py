"""
Process the raw Mitchell datacubes into a format more accessible
for binning and fitting.

The changes are:
 - Coordinates converted from (RA, DEC) to projected Cartesian arcsec
 - Arc frames are fit by Gaussian line profiles and replaced with the
   resulting samples of fwhm(lambda) for each fiber

input:
  takes one command line argument, a path to the input parameter text file
  s1_process_mitchell_params_example.txt is an example
  can take multiple parameter files if you want to process multiple galaxies
  (give one param file per galaxy)

output:
  one processed datacube for each input raw datacube
  one pdf with diagnostic plots
"""


import argparse
import re
import os
import functools

import numpy as np
import pandas as pd
import shapely.geometry as geo
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import astropy.io.fits as fits

import utilities as utl
import massivepy.constants as const
import massivepy.spectralresolution as res
import massivepy.IFUspectrum as ifu
import massivepy.io as mpio
import massivepy.gausshermite as gh

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
    raw_cube_path = input_params['raw_mitchell_cube']
    target_positions = pd.read_csv(input_params["target_positions"],
                                   comment='#', sep="[ \t]+",
                                   engine='python')
    run_name = input_params['run_name']
    # construct output file names
    output_path_maker = lambda f, ext: os.path.join(output_dir,
                            "{}-s1-{}-{}.{}".format(gal_name, run_name, f, ext))
    plot_path = output_path_maker('fibermaps','pdf')
    ir_path = output_path_maker('ir','txt')
    # save relevant info for plotting to a dict
    plot_info = {'plot_path': plot_path,'ir_path': ir_path,
                 'raw_cube_path': raw_cube_path,
                 'targets_path': input_params['target_positions'],
                 'gal_name':gal_name}
    things_to_plot.append(plot_info)

    # decide whether to continue with script or skip to plotting
    if os.path.isfile(ir_path):
        if input_params['skip_rerun']=='yes':
            print '\nSkipping re-run of {}, plotting only'.format(gal_name)
            continue
        elif input_params['skip_rerun']=='no':
            print '\nRunning {} again, will overwrite output'.format(gal_name)
        else:
            raise Exception("skip_rerun must be yes or no")

    # start processing
    ifuset, arcs = ifu.read_raw_datacube(raw_cube_path,
                                         input_params['target_positions'],
                                         gal_name,
                                         return_arcs=True)
    gal_center_ra = ifuset.coord_comments['galaxy center RA']
    gal_center_dec = ifuset.coord_comments['galaxy center DEC']
    gal_pa = ifuset.coord_comments['galaxy pa']
    print "\n{}".format(gal_name)
    print "  raw datacube: {}".format(raw_cube_path)
    print "        center: {}, {}".format(gal_center_ra,gal_center_dec)
    print "            pa: {}".format(gal_pa)
    redshift = ifuset.spectrumset.comments['redshift']
    inst_waves = ifuset.spectrumset.waves*(1+redshift)
    print "  fitting arc frames..."
    spec_res_samples = res.fit_arcset(inst_waves, arcs,
                                      const.mitchell_arc_centers,
                                      const.mitchell_nominal_spec_resolution)
    nfibers, nlines, ncols = spec_res_samples.shape
    ir_header = "Fits of arc frames for each fiber"
    ir_header += "\n interpolated from {} arc lamp lines".format(nlines)
    ir_header += "\n reported in instrument rest frame"
    ir_header += "\nFour columns for each line, as follows:"
    ir_header += "\n fiducial center, fit center, fit fwhm, fit height"
    ir_savearray = np.zeros((nfibers, 4*nlines))
    fmt = nlines*['%-7.6g','%-7.6g','%-7.4g','%-7.4g']
    ir_savearray[:,0::4] = const.mitchell_arc_centers
    ir_savearray[:,1::4] = spec_res_samples[:,:,0]
    ir_savearray[:,2::4] = spec_res_samples[:,:,1]
    ir_savearray[:,3::4] = spec_res_samples[:,:,2]
    np.savetxt(ir_path, ir_savearray, fmt=fmt, delimiter='\t', header=ir_header)
    print "  saved ir to text file."

for plot_info in things_to_plot:
    # read data, with the goal of not referencing ifuset after this section
    ifuset, arcs = ifu.read_raw_datacube(plot_info['raw_cube_path'],
                                         plot_info['targets_path'],
                                         plot_info['gal_name'],
                                         ir_path=plot_info['ir_path'],
                                         return_arcs=True)
    # set up fiber information
    fiberids = ifuset.spectrumset.ids
    fibersize = ifuset.linear_scale
    xcoords = -ifuset.coords[:,0] # flip from +x = east to +x = west
    ycoords = ifuset.coords[:,1]
    squaremax = np.amax(np.abs(ifuset.coords)) + fibersize
    rcoords = np.sqrt(xcoords**2 + ycoords**2)
    coordunit = ifuset.coords_unit
    # set up flux
    rawflux = ifuset.spectrumset.compute_flux()
    fluxes = np.where(rawflux>0,rawflux,max(rawflux)*np.ones(rawflux.shape))
    fluxmin = min(fluxes)
    fluxmax = max(fluxes)
    fluxunit = ifuset.spectrumset.integratedflux_unit
    cmap_flux = 'Reds'
    norm_flux = mpl.colors.LogNorm(vmin=fluxmin,vmax=fluxmax)
    mappable_flux = plt.cm.ScalarMappable(cmap=cmap_flux, norm=norm_flux)
    mappable_flux.set_array([fluxmin,fluxmax])
    fluxcolors = mappable_flux.to_rgba(fluxes)
    # set up s2n
    raws2n = ifuset.spectrumset.compute_mean_s2n()
    s2n = np.where(raws2n>0,raws2n,max(raws2n)*np.ones(raws2n.shape))
    s2nmin = min(s2n)
    s2nmax = max(s2n)
    cmap_s2n = 'Greens'
    norm_s2n = mpl.colors.LogNorm(vmin=s2nmin,vmax=s2nmax)
    mappable_s2n = plt.cm.ScalarMappable(cmap=cmap_s2n, norm=norm_s2n)
    mappable_s2n.set_array([s2nmin,s2nmax])
    s2ncolors = mappable_s2n.to_rgba(s2n)
    # set up spectra and arc info
    skipnumber = 1
    waves = ifuset.spectrumset.waves
    dwave = waves[-1] - waves[0]
    inst_waves = waves*(1+ifuset.spectrumset.comments['redshift'])
    spectra = ifuset.spectrumset.spectra[::skipnumber]
    arcspectra = arcs[::skipnumber]
    arcspectra = (arcspectra.T/np.max(arcspectra, axis=1)).T

    # load up the ir textfile
    ir_textinfo = np.genfromtxt(plot_info['ir_path'])
    ir_fidcenter = ir_textinfo[::skipnumber,0::4]
    ir_center = ir_textinfo[::skipnumber,1::4]
    ir_fwhm = ir_textinfo[::skipnumber,2::4]
    ir_height = ir_textinfo[::skipnumber,3::4]
    nskipfibers, nlines = ir_fidcenter.shape

    ### plotting begins ###
    pdf = PdfPages(plot_info['plot_path'])

    # do flux, s2n maps and flux, s2n vs radius
    figs = {}
    axs = {}
    fignames = ['fluxmap','s2nmap','fluxrad','s2nrad']
    figtitles = ['Flux map','s2n map','Flux vs radius','s2n vs radius']
    for figname, figtitle in zip(fignames,figtitles):
        figs[figname] = plt.figure(figsize=(6,6))
        figs[figname].suptitle(figtitle)
        axs[figname] = figs[figname].add_axes([0.15,0.1,0.7,0.7])
    for ifiber,fiberid in enumerate(fiberids):
        x,y,r = xcoords[ifiber],ycoords[ifiber],rcoords[ifiber]
        patch = functools.partial(patches.Circle,(x,y),fibersize,lw=0.25)
        txtkw = {'fontsize':5,
                 'horizontalalignment':'center',
                 'verticalalignment':'center'}
        axs['fluxmap'].add_patch(patch(fc=fluxcolors[ifiber]))
        axs['fluxmap'].text(x,y,str(fiberid),**txtkw)
        axs['s2nmap'].add_patch(patch(fc=s2ncolors[ifiber]))
        axs['s2nmap'].text(x,y,str(fiberid),**txtkw)
        axs['fluxrad'].text(r,np.log10(fluxes[ifiber]),str(fiberid),**txtkw)
        axs['s2nrad'].text(r,np.log10(s2n[ifiber]),str(fiberid),**txtkw)
    axs['fluxmap'].axis([-squaremax,squaremax,-squaremax,squaremax])
    axs['s2nmap'].axis([-squaremax,squaremax,-squaremax,squaremax])
    axs['fluxrad'].axis([min(rcoords),max(rcoords),
                         np.log10(fluxmin),np.log10(fluxmax)])
    axs['s2nrad'].axis([min(rcoords),max(rcoords),
                        np.log10(s2nmin),np.log10(s2nmax)])
    label_x = r'$\leftarrow$east ({}) west$\rightarrow$'.format(coordunit)
    label_y = r'$\leftarrow$south ({}) north$\rightarrow$'.format(coordunit)
    label_r = r'radius ({})'.format(coordunit)
    label_flux = r'flux (log 10 [{}])'.format(fluxunit)
    label_s2n = r's2n (log 10)'
    axs['fluxmap'].set_xlabel(label_x)
    axs['fluxmap'].set_ylabel(label_y)
    axs['s2nmap'].set_xlabel(label_x)
    axs['s2nmap'].set_ylabel(label_y)
    axs['fluxrad'].set_xlabel(label_r)
    axs['fluxrad'].set_ylabel(label_flux)
    axs['s2nrad'].set_xlabel(label_r)
    axs['s2nrad'].set_ylabel(label_s2n)
    # do colorbars
    for fig,m,l in zip([figs['fluxmap'],figs['s2nmap']],
                       [mappable_flux,mappable_s2n],
                       [label_flux,label_s2n]):
        axC = fig.add_axes([0.15,0.8,0.7,0.8])
        axC.set_visible(False)
        cb = fig.colorbar(m,ax=axC,label=l,orientation='horizontal',
                          ticks=mpl.ticker.LogLocator(subs=range(10)))
    for fn in fignames:
        pdf.savefig(figs[fn])
        plt.close(figs[fn])

    # plot all fiber spectra
    fig = plt.figure(figsize=(6,6))
    fig.suptitle('fiber spectra')
    ax = fig.add_axes([0.15,0.1,0.7,0.7])
    for i in range(nskipfibers):
        ax.plot(waves,spectra[i],c='k',alpha=0.1)
    ax.plot(waves,0*waves,c='r')
    specmax = np.percentile(spectra,99.99)
    ax.axis([waves[0],waves[-1],-0.2*specmax,1.2*specmax])
    ax.set_xlabel('wavelength')
    ax.set_rasterized(True) # works, is ugly, might want to bump dpi up
    pdf.savefig(fig)
    plt.close(fig)

    # plot the ir vs wavelength for each fiber
    fig = plt.figure(figsize=(6,6))
    fig.suptitle('fiber ir')
    ax = fig.add_axes([0.15,0.1,0.7,0.7])
    fiber_cmap = plt.cm.get_cmap('cool')
    for i in range(nskipfibers):
        ax.plot(ir_center[i,:],ir_fwhm[i,:],c=fiber_cmap(i/float(nskipfibers)),
                alpha=0.2,rasterized=True)
    axC = fig.add_axes([0.15,0.8,0.7,0.8])
    axC.set_visible(False)
    mappable_bins = plt.cm.ScalarMappable(cmap=fiber_cmap)
    mappable_bins.set_array([0,nskipfibers])
    fig.colorbar(mappable_bins,orientation='horizontal',ax=axC,
                 label='fiber number')
    ax.set_xlabel('wavelength')
    ax.set_ylabel('fwhm')
    ax.set_rasterized(True) # works, is ugly, might want to bump dpi up
    pdf.savefig(fig)
    plt.close(fig)

    # plot the arc lines
    fig = plt.figure(figsize=(6,6))
    fig.suptitle('arc spectra')
    ax = fig.add_axes([0.15,0.1,0.7,0.7])
    for i in range(nskipfibers):
        ax.plot(inst_waves,arcspectra[i],c='k',alpha=0.2,lw=1.2)
    for i in range(nlines):
        ax.axvline(ir_fidcenter[0,i],c='r',lw=0.8)
    ax.axis([inst_waves[0],inst_waves[-1],
             -0.2*np.max(arcspectra),np.max(arcspectra)])
    ax.set_rasterized(True) # works, is ugly, might want to bump dpi up
    pdf.savefig(fig)
    plt.close(fig)

    # do a zoom-in of each arc line to check that it fits
    fig = plt.figure(figsize=(6,nlines+2))
    fig.suptitle('zoom in of each arc line fit')
    ax = fig.add_axes([0.05,0.05,0.9,0.9])
    ir_avg = np.average(ir_fwhm)
    for i in range(nlines):
        for j in range(nskipfibers):
            startwave = ir_fidcenter[j,i] - 2*ir_avg
            endwave = ir_fidcenter[j,i] + 2*ir_avg
            startpix,endpix = np.searchsorted(inst_waves,[startwave,endwave])
            waves_offset = inst_waves[startpix:endpix] - ir_fidcenter[j,i]
            ax.plot(waves_offset,i + arcspectra[j][startpix:endpix],
                    c='k',alpha=0.2)
            params=[ir_center[j,i] - ir_fidcenter[j,i],
                    ir_fwhm[j,i]/const.gaussian_fwhm_over_sigma]
            model = gh.unnormalized_gausshermite_pdf(waves_offset,params)
            model = model*max(arcspectra[j][startpix:endpix])/max(model)
            ax.plot(waves_offset,i + model,c='b',alpha=0.2)
            ax.vlines(params[0]+0.5*params[1],i,i+1,color='y',alpha=0.2)
            ax.vlines(params[0]-0.5*params[1],i,i+1,color='y',alpha=0.2)
            ax.vlines(params[0],i,i+1,color='g',alpha=0.2)
        ax.text(ir_avg,i+0.2,"{:.4f}".format(np.average(ir_center[:,i])),
                color='g',horizontalalignment='left')
        ax.text(-ir_avg,i+0.2,"{:.4f}".format(np.average(ir_fidcenter[:,i])),
                color='k',horizontalalignment='right')
    ax.axvline(c='k')
    ax.axis([-2*ir_avg,2*ir_avg,0,nlines])
    ax.set_xlabel('wavelength offset')
    ax.set_yticklabels([])
    ax.set_rasterized(True) # works, is ugly, might want to bump dpi up
    pdf.savefig(fig)
    plt.close(fig)

    pdf.close()
