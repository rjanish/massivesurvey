"""
MASSIVE-specific plotting routines:

This file contains the main plotting fuction for s1_process_mitchell.
"""

import functools

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches

import massivepy.constants as const
import massivepy.spectralresolution as res
import massivepy.IFUspectrum as ifu
import massivepy.gausshermite as gh
import massivepy.plotting as mplt


def plot_s1_process_mitchell(gal_name=None,raw_cube_path=None,
                             targets_path=None,ir_path=None,plot_path=None):
    # read data, with the goal of not referencing ifuset after this section
    ifuset, arcs = ifu.read_raw_datacube(raw_cube_path,targets_path,gal_name,
                                         ir_path=ir_path,return_arcs=True)
    ir = res.read_specres(ir_path)

    # set up fiber coordinate information
    fiberids = ifuset.spectrumset.ids
    fibersize = ifuset.linear_scale
    xcoords = -ifuset.coords[:,0] # flip from +x = east to +x = west
    ycoords = ifuset.coords[:,1]
    squaremax = np.amax(np.abs(ifuset.coords)) + fibersize
    rcoords = np.sqrt(xcoords**2 + ycoords**2)
    coordunit = ifuset.coords_unit 
    label_x = r'$\leftarrow$east ({}) west$\rightarrow$'.format(coordunit)
    label_y = r'$\leftarrow$south ({}) north$\rightarrow$'.format(coordunit)
    label_r = r'radius ({})'.format(coordunit)

    # set up colormaps for total flux, s2n
    fluxunit = ifuset.spectrumset.integratedflux_unit
    fluxcolors = mplt.colormap_setup(ifuset.spectrumset.compute_flux(),
                                cmap='Reds',logsafe='max')
    s2ncolors = mplt.colormap_setup(ifuset.spectrumset.compute_mean_s2n(),
                               cmap='Greens',logsafe='max')
    label_flux = r'flux (log 10 [{}])'.format(fluxunit)
    label_s2n = r's2n (log 10)'

    # set up spectra and arc info, optionally skipping some fibers
    skipnumber = 100
    waves = ifuset.spectrumset.waves
    dwave = waves[-1] - waves[0]
    inst_waves = waves*(1+ifuset.spectrumset.comments['redshift'])
    spectra = ifuset.spectrumset.spectra[::skipnumber]
    arcspectra = arcs[::skipnumber]
    arcspectra = (arcspectra.T/np.max(arcspectra, axis=1)).T
    ir = ir[::skipnumber,:]
    nskipfibers, nlines = ir.shape


    ### plotting begins ###
    pdf = PdfPages(plot_path)

    # do flux and s2n fiber maps
    fig1, ax1 = mplt.scalarmap(figtitle='flux map',
                               xlabel=label_x, ylabel=label_y,
                               axC_mappable=fluxcolors['mappable'],
                               axC_label=label_flux)
    fig2, ax2 = mplt.scalarmap(figtitle='s2n map',
                               xlabel=label_x, ylabel=label_y,
                               axC_mappable=s2ncolors['mappable'],
                               axC_label=label_s2n)
    for ifiber,fiberid in enumerate(fiberids):
        x, y = xcoords[ifiber], ycoords[ifiber]
        patch = functools.partial(patches.Circle,(x,y),fibersize,lw=0.25)
        txtkw = {'fontsize':5,
                 'horizontalalignment':'center',
                 'verticalalignment':'center'}
        ax1.add_patch(patch(fc=fluxcolors['c'][ifiber]))
        ax1.text(x,y,str(fiberid),**txtkw)
        ax2.add_patch(patch(fc=s2ncolors['c'][ifiber]))
        ax2.text(x,y,str(fiberid),**txtkw)
    ax1.axis([-squaremax,squaremax,-squaremax,squaremax])
    ax2.axis([-squaremax,squaremax,-squaremax,squaremax])
    for fig in [fig1, fig2]:
        pdf.savefig(fig)
        plt.close(fig)

    # do flux and s2n vs radius
    fig1, ax1 = mplt.scalarmap(figtitle='Flux vs radius',
                               xlabel=label_r, ylabel = label_flux)
    fig2, ax2 = mplt.scalarmap(figtitle='s2n vs radius',
                               xlabel=label_r, ylabel = label_s2n)
    for ifiber,fiberid in enumerate(fiberids):
        r = rcoords[ifiber]
        txtkw = {'fontsize':5,
                 'horizontalalignment':'center',
                 'verticalalignment':'center'}
        ax1.text(r,fluxcolors['x_norm'][ifiber],str(fiberid),**txtkw)
        ax2.text(r,s2ncolors['x_norm'][ifiber],str(fiberid),**txtkw)
    ax1.axis([min(rcoords),max(rcoords),
              fluxcolors['vmin_norm'],fluxcolors['vmax_norm']])
    ax2.axis([min(rcoords),max(rcoords),
                        s2ncolors['vmin_norm'],s2ncolors['vmax_norm']])
    for fig in [fig1, fig2]:
        pdf.savefig(fig)
        plt.close(fig)

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
        ax.plot(ir['fitcenter'][i,:],ir['fwhm'][i,:],c=fiber_cmap(i/float(nskipfibers)),
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
        ax.axvline(ir['center'][0,i],c='r',lw=0.8)
    ax.axis([inst_waves[0],inst_waves[-1],
             -0.2*np.max(arcspectra),np.max(arcspectra)])
    ax.set_rasterized(True) # works, is ugly, might want to bump dpi up
    pdf.savefig(fig)
    plt.close(fig)

    # do a zoom-in of each arc line to check that it fits
    fig = plt.figure(figsize=(6,nlines+2))
    fig.suptitle('zoom in of each arc line fit')
    ax = fig.add_axes([0.05,0.05,0.9,0.9])
    ir_avg = np.average(ir['fwhm'])
    for i in range(nlines):
        for j in range(nskipfibers):
            startwave = ir['center'][j,i] - 2*ir_avg
            endwave = ir['center'][j,i] + 2*ir_avg
            startpix,endpix = np.searchsorted(inst_waves,[startwave,endwave])
            waves_offset = inst_waves[startpix:endpix] - ir['center'][j,i]
            ax.plot(waves_offset,i + arcspectra[j][startpix:endpix],
                    c='k',alpha=0.2)
            params=[ir['fitcenter'][j,i] - ir['center'][j,i],
                    ir['fwhm'][j,i]/const.gaussian_fwhm_over_sigma]
            model = gh.unnormalized_gausshermite_pdf(waves_offset,params)
            model = model*max(arcspectra[j][startpix:endpix])/max(model)
            ax.plot(waves_offset,i + model,c='b',alpha=0.2)
            ax.vlines(params[0]+0.5*params[1],i,i+1,color='y',alpha=0.2)
            ax.vlines(params[0]-0.5*params[1],i,i+1,color='y',alpha=0.2)
            ax.vlines(params[0],i,i+1,color='g',alpha=0.2)
        ax.text(ir_avg,i+0.2,"{:.4f}".format(np.average(ir['fitcenter'][:,i])),
                color='g',horizontalalignment='left')
        ax.text(-ir_avg,i+0.2,"{:.4f}".format(np.average(ir['center'][:,i])),
                color='k',horizontalalignment='right')
    ax.axvline(c='k')
    ax.axis([-2*ir_avg,2*ir_avg,0,nlines])
    ax.set_xlabel('wavelength offset')
    ax.set_yticklabels([])
    ax.set_rasterized(True) # works, is ugly, might want to bump dpi up
    pdf.savefig(fig)
    plt.close(fig)

    pdf.close()
    return
