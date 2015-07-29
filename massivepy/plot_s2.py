"""
MASSIVE-specific plotting routines:

This file contains the main plotting fuction for s2_bin_mitchell.
"""

import functools

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
import descartes

import massivepy.constants as const
#import massivepy.spectralresolution as res
import massivepy.IFUspectrum as ifu
import massivepy.spectrum as spec
#import massivepy.gausshermite as gh
#import massivepy.plotting as mplt
import plotting.geo_utils as geo_utils


def plot_s2_bin_mitchell(plot_info):
    plot_path = plot_info['plot_path']
    fiberids, binids = np.genfromtxt(plot_info['fiberinfo_path'],
                                     dtype=int,unpack=True)
    bininfo = np.genfromtxt(plot_info['bininfo_path'],names=True,skip_header=1)
    binsettings = open(plot_info['bininfo_path'],'r').readlines()[18]
    aspect_ratio, s2n_threshold = binsettings.strip().split()[1:]
    aspect_ratio, s2n_threshold = eval(aspect_ratio[:-1]), eval(s2n_threshold)
    nbins = len(bininfo)
    # these lines are a terrible idea.
    ma_line = open(plot_info['bininfo_path'],'r').readlines()[9]
    ma_theta = np.pi/2 + np.deg2rad(float(ma_line.strip().split()[-1]))

    ifuset = ifu.read_raw_datacube(plot_info['raw_cube_path'],
                                   plot_info['targets_path'],
                                   plot_info['gal_name'],
                                   ir_path=plot_info['ir_path'])
    ifuset.crop(plot_info['crop_region'])
    fiber_coords = ifuset.coords.copy()
    coordunit = ifuset.coords_unit
    fibersize = const.mitchell_fiber_radius.value #Assuming units match!
    fiber_coords[:, 0] *= -1  # east-west reflect
    squaremax = np.amax(np.abs(ifuset.coords)) + fibersize

    specset = spec.read_datacube(plot_info['binspectra_path'])
    # use colorbar limits from fiber maps, for continuity
    rawfiberfluxes = ifuset.spectrumset.compute_flux()
    fiberfluxes = np.where(rawfiberfluxes>0, rawfiberfluxes,
                           max(rawfiberfluxes)*np.ones(rawfiberfluxes.shape))
    fluxmin = min(fiberfluxes)
    fluxmax = max(fiberfluxes)
    fluxunit = specset.integratedflux_unit
    cmap_flux = 'Reds'
    bin_s2n = specset.compute_mean_s2n()
    rawfiber_s2n = ifuset.spectrumset.compute_mean_s2n()
    fiber_s2n = np.where(rawfiber_s2n>0, rawfiber_s2n,
                         max(rawfiber_s2n)*np.ones(rawfiber_s2n.shape))
    s2nmin = min(fiber_s2n)
    s2nmax = max(fiber_s2n)
    s2nmin_bin = min(bin_s2n)
    s2nmax_bin = max(bin_s2n)
    cmap_s2n = 'Greens'

    specset_full = spec.read_datacube(plot_info['fullbin_path'])

    ### plotting begins ###
    pdf = PdfPages(plot_path)

    # plot bin map, bin flux, bin s2n (two versions), bin centers comparison
    figs = {}
    axs = {}
    fignames = ['map','flux','s2n','s2nbin','centers']
    figtitles = ['Bin map (s2n {}, ar {})'.format(s2n_threshold, aspect_ratio),
                 'Bin flux map','Bin s2n map','Bin s2n map rescaled',
                 'Bin centers comparison (cartesian vs polar)']
    for figname, figtitle in zip(fignames,figtitles):
        figs[figname] = plt.figure(figsize=(6,6))
        figs[figname].suptitle(figtitle)
        axs[figname] = figs[figname].add_axes([0.15,0.1,0.7,0.7])
    # prep bin coloring (arbitrary colors, flux colormap, and s2n colormaps)
    mycolors = ['b','g','c','m','r','y']
    bincolors = {}
    for binid in set(binids):
        bincolors[binid] = mycolors[binid % len(mycolors)]
    bincolors[const.badfiber_bin_id] = 'k'
    bincolors[const.unusedfiber_bin_id] = '0.7'
    norm_flux = mpl.colors.LogNorm(vmin=fluxmin,vmax=fluxmax)
    mappable_flux = plt.cm.ScalarMappable(cmap=cmap_flux, norm=norm_flux)
    mappable_flux.set_array([fluxmin,fluxmax])
    fluxcolors = mappable_flux.to_rgba(bininfo['flux'])
    norm_s2n = mpl.colors.LogNorm(vmin=s2nmin,vmax=s2nmax)
    mappable_s2n = plt.cm.ScalarMappable(cmap=cmap_s2n, norm=norm_s2n)
    mappable_s2n.set_array([s2nmin,s2nmax])
    s2ncolors = mappable_s2n.to_rgba(bin_s2n)
    norm_s2nbin = mpl.colors.LogNorm(vmin=s2nmin_bin,vmax=s2nmax_bin)
    mappable_s2nbin = plt.cm.ScalarMappable(cmap=cmap_s2n, norm=norm_s2nbin)
    mappable_s2nbin.set_array([s2nmin_bin,s2nmax_bin])
    s2nbincolors = mappable_s2nbin.to_rgba(bin_s2n)
    # loop over fibers
    for fiber_id,bin_id in zip(fiberids,binids):
        axs['map'].add_patch(patches.Circle(fiber_coords[fiber_id,:],fibersize,
                                     fc=bincolors[bin_id],ec='none',alpha=0.8))
    # loop over bins
    for bin_iter,bin_id in enumerate(bininfo['binid']):
        bincolor = bincolors[int(bin_id)]
        # draw bin number at bin center
        xbin=-bininfo['r'][bin_iter]*np.sin(np.deg2rad(bininfo['th'][bin_iter]))
        ybin=bininfo['r'][bin_iter]*np.cos(np.deg2rad(bininfo['th'][bin_iter]))
        axs['map'].plot(xbin,ybin,ls='',marker='o',mew=1.0,ms=8.0,mec='k',
                        mfc=bincolor)
        axs['map'].text(xbin-0.2,ybin-0.1,str(int(bin_id)),fontsize=5,
                 horizontalalignment='center',verticalalignment='center')
        # draw bin center, both versions
        axs['centers'].plot(bininfo['x'][bin_iter],bininfo['y'][bin_iter],
                            ls='',marker='s',mew=0,ms=5.0,mfc='r')
        axs['centers'].plot(xbin,ybin,ls='',marker='o',mew=0,ms=5.0,mfc='k')
        # draw bin outline and flux/s2n maps
        if not np.isnan(bininfo['rmin'][bin_iter]):
            thmin = 90 + bininfo['thmin'][bin_iter]
            thmax = 90 + bininfo['thmax'][bin_iter]
            bin_poly = geo_utils.polar_box(bininfo['rmin'][bin_iter], 
                                           bininfo['rmax'][bin_iter],
                                           thmin,thmax)
            # also do a transparent fill in bincolor to make sure bins match
            # if the storage of bin boundaries breaks, this will help notice
            patch = functools.partial(descartes.PolygonPatch,bin_poly,lw=1.5)
            axs['map'].add_patch(patch(fc=bincolor,
                                       ec='none',alpha=0.5,zorder=-1))
            axs['map'].add_patch(patch(fc='none'))
            axs['flux'].add_patch(patch(fc=fluxcolors[bin_iter]))
            axs['s2n'].add_patch(patch(fc=s2ncolors[bin_iter]))
            axs['s2nbin'].add_patch(patch(fc=s2nbincolors[bin_iter]))
            axs['centers'].add_patch(patch(fc='none'))
        else:
            patch = functools.partial(patches.Circle,(bininfo['x'][bin_iter],
                                    bininfo['y'][bin_iter]),fibersize,lw=0.25)
            axs['flux'].add_patch(patch(fc=fluxcolors[bin_iter]))
            axs['s2n'].add_patch(patch(fc=s2ncolors[bin_iter]))
            axs['s2nbin'].add_patch(patch(fc=s2nbincolors[bin_iter]))

    label_x = r'$\leftarrow$east ({}) west$\rightarrow$'.format(coordunit)
    label_y = r'$\leftarrow$south ({}) north$\rightarrow$'.format(coordunit)
    label_flux = r'flux [{}]'.format(fluxunit)
    label_s2n = r's2n'
    # do colorbars
    for fig,m,l in zip([figs['flux'],figs['s2n'],figs['s2nbin']],
                       [mappable_flux,mappable_s2n,mappable_s2nbin],
                       [label_flux,label_s2n,label_s2n]):
        axC = fig.add_axes([0.15,0.8,0.7,0.8])
        axC.set_visible(False)
        cb = fig.colorbar(m,ax=axC,label=l,orientation='horizontal',
                          ticks=mpl.ticker.LogLocator(subs=range(10)))
        # do some annoying fiddling with ticks to get minor ticks, end labels
        ticks = m.norm.inverse(cb.ax.xaxis.get_majorticklocs())
        cb.set_ticks(ticks) # for some reason, required before setting labels
        ticklabels = ['' for t in ticks]
        ticklabels[0] = ticks[0]
        ticklabels[-1] = ticks[-1]
        cb.set_ticklabels(ticklabels)
    # draw ma, set labels, save and close figures
    rmax = np.nanmax(bininfo['rmax'])
    for fn in fignames:
        axs[fn].plot([-rmax*1.1*np.cos(ma_theta), rmax*1.1*np.cos(ma_theta)],
                     [-rmax*1.1*np.sin(ma_theta), rmax*1.1*np.sin(ma_theta)],
                     linewidth=1.5, color='r')
        axs[fn].axis([-squaremax,squaremax,-squaremax,squaremax])
        axs[fn].set_xlabel(label_x)
        axs[fn].set_ylabel(label_y)
        pdf.savefig(figs[fn])
        plt.close(figs[fn])


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
    fibertobindict = {f:b for (f,b) in zip(fiberids,binids)}
    rcoords = np.sqrt(fiber_coords[:,0]**2 + fiber_coords[:,1]**2)
    # reuse the flux color mapping from above on the fiber fluxes
    fluxcolors = mappable_flux.to_rgba(fiberfluxes)
    for ifiber in range(len(fiberfluxes)):
        fiber_id = ifuset.spectrumset.ids[ifiber]
        bin_id = fibertobindict[fiber_id]
        logflux = np.log10(fiberfluxes[ifiber])
        if not bin_id==const.badfiber_bin_id:
            ax1.add_patch(patches.Circle(fiber_coords[ifiber,:],
                                         fibersize,lw=0.25,
                                         fc=fluxcolors[ifiber]))
            ax2.text(rcoords[ifiber],logflux,str(fiber_id),fontsize=5,
                     horizontalalignment='center',verticalalignment='center')
            ax3.text(rcoords[ifiber],logflux,str(fiber_id),fontsize=5,
                     horizontalalignment='center',verticalalignment='center',
                     alpha=0.3)
        else:
            ax3.text(rcoords[ifiber],logflux,str(fiber_id),fontsize=5,
                     horizontalalignment='center',verticalalignment='center')
            ax3.plot(rcoords[ifiber],logflux,ls='',marker='o',
                     mec='r',mfc='none',ms=10,lw=1.0)
        ax1.text(fiber_coords[ifiber,0],fiber_coords[ifiber,1],
                 str(fiber_id),fontsize=5,
                 horizontalalignment='center',verticalalignment='center')
    ax1.axis([-squaremax,squaremax,-squaremax,squaremax])
    ax2.axis([min(rcoords),max(rcoords),np.log10(fluxmin),np.log10(fluxmax)])
    ax3.axis([min(rcoords),max(rcoords),np.log10(fluxmin),np.log10(fluxmax)])
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
    return
