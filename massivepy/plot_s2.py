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
import massivepy.IFUspectrum as ifu
import massivepy.spectrum as spec
import massivepy.binning as binning
import massivepy.plot_massive as mplt
import massivepy.io as mpio
from plotting.geo_utils import polar_box


def plot_s2_bin_mitchell(gal_name=None,plot_path=None,raw_cube_path=None,
                         targets_path=None,ir_path=None,fiberinfo_path=None,
                         bininfo_path=None,binspectra_path=None,
                         crop_region=None,fullbin_path=None):
    # read fiber and bin info
    fiberids, binids = np.genfromtxt(fiberinfo_path,
                                     dtype=int,unpack=True)
    fibermeta = mpio.read_friendly_header(fiberinfo_path)
    gal_info = mpio.get_gal_info(targets_path,gal_name)
    bindata, binmeta = binning.read_bininfo(bininfo_path)
    nbins = len(bindata)
    # set up bin colors
    mycolors = ['b','g','c','m','r','y']
    bincolors = {}
    for binid in set(binids):
        bincolors[binid] = mycolors[binid % len(mycolors)]
    bincolors[const.badfiber_bin_id] = 'k'
    bincolors[const.unusedfiber_bin_id] = '0.7'
    # this is not super elegant, but oh well
    goodfibers = []
    for fiberid,binid in zip(fiberids,binids):
        if not binid==const.badfiber_bin_id:
            goodfibers.append(fiberid)
    ircolors = mplt.lin_colormap_setup(bindata['binid'],cmap='cool')
    label_ir = 'fwhm in wavelength units'

    # read spectra fits files
    # (can probably ditch reading the ir, won't actually use it)
    ifuset = ifu.read_raw_datacube(raw_cube_path,gal_info,gal_name) ##J##
    ifuset.crop(crop_region) # everything plotted here is the cropped version!
    ifuset2 = ifuset.get_subset(goodfibers)
    specset = spec.read_datacube(binspectra_path)
    specset_full = spec.read_datacube(fullbin_path)
    label_waves = r'wavelength (angstroms assumed)'

    # set up fiber coordinate information
    if not len(fiberids)==len(ifuset.spectrumset.ids):
        raise Exception("Fiber ids from raw cube don't match ids in fiberinfo")
    fibersize = ifuset.linear_scale
    fiber_coords = ifuset.coords.copy()
    fiber_coords[:,0] *= -1 # flip from +x = east to +x = west
    rcoords = np.sqrt(fiber_coords[:,0]**2 + fiber_coords[:,1]**2)
    squaremax = np.amax(np.abs(ifuset.coords)) + fibersize
    coordunit = ifuset.coords_unit
    label_x = r'$\leftarrow$east ({}) west$\rightarrow$'.format(coordunit)
    label_y = r'$\leftarrow$south ({}) north$\rightarrow$'.format(coordunit)
    label_r = r'radius ({})'.format(coordunit)

    # set up colormaps for bin flux, s2n (don't use logsafe, shouldn't need it)
    # note this should be able to use specset.compute_flux(), BUTBUTBUT
    #  the spectra in binspectra.fits are still normalized arbitrarily
    binfluxcolors = mplt.colormap_setup(bindata['flux'],cmap='Reds')
    bins2ncolors = mplt.colormap_setup(specset.compute_mean_s2n(),cmap='Greens')
    fiberfluxcolors = mplt.colormap_setup(ifuset.spectrumset.compute_flux(),
                                          cmap='Reds',logsafe='max')
    fiberfluxcolors2 = mplt.colormap_setup(ifuset2.spectrumset.compute_flux(),
                                           cmap='Reds',logsafe='max')
    fibers2ncolors = mplt.colormap_setup(ifuset.spectrumset.compute_mean_s2n(),
                                         cmap='Greens',logsafe='max')
    fibers2ncolors2 =mplt.colormap_setup(ifuset2.spectrumset.compute_mean_s2n(),
                                         cmap='Greens',logsafe='max')
    label_flux = r'flux [{}]'.format(specset.integratedflux_unit)
    label_s2n = r's2n'

    # set up fiber spectra to plot, optionally skipping some for speed
    skipnumber = 1
    plotfibers = ifuset.spectrumset.ids[::skipnumber]
    fiberwaves = ifuset.spectrumset.waves
    fiberspectra = ifuset.spectrumset.spectra[::skipnumber]
    fiberspectra[ifuset.spectrumset.metaspectra['bad_data'][::skipnumber]]=-1000



    ### plotting begins ###
    pdf = PdfPages(plot_path)



    # plot bin maps, bin fluxes/s2n, bin centers comparison
    fig_keys = ['map','flux','s2n','cent','fmap','fmap2','smap','smap2']
    titles = [('{} Bin map (s2n {}, ar {})'.format(gal_name,
                            binmeta['threshold s2n'],binmeta['threshold ar'])),
              '{} Bin flux map'.format(gal_name),
              '{} Bin s2n map'.format(gal_name),
              '{} Bin centers (cartesian v polar)'.format(gal_name),
              '{} Fiber flux map (with cropping)'.format(gal_name),
              ('{} Fiber flux map (with cropping and fiber removal)'
               ''.format(gal_name)),
              '{} Fiber s2n map (with cropping)'.format(gal_name),
              ('{} Fiber s2n map (with cropping and fiber removal)'
               ''.format(gal_name))]
    mappables = [None,binfluxcolors['mappable'],bins2ncolors['mappable'],None,
                 fiberfluxcolors['mappable'],fiberfluxcolors2['mappable'],
                 fibers2ncolors['mappable'],fibers2ncolors2['mappable']]
    mlabels = [None,label_flux,label_s2n,None] + 2*[label_flux] + 2*[label_s2n]
    figs,axs = {}, {}
    # create figures and axes
    for i,k in enumerate(fig_keys):
        figs[k],axs[k] = mplt.scalarmap(figtitle=titles[i],
                                        xlabel=label_x,ylabel=label_y,
                                        axC_mappable=mappables[i],
                                        axC_label=mlabels[i])
    # draw cartesian bin centers
    axs['cent'].plot(bindata['x'],bindata['y'],ls='',marker='s',mew=0,ms=5.0,
                     mfc='r')
    # loop over fibers
    txtkw = {'horizontalalignment':'center','verticalalignment':'center',
             'fontsize':5}
    i2 = 0 # a hacky way to access only the fluxes for good fibers
    for ifiber, (fiber_id,bin_id) in enumerate(zip(fiberids,binids)):
        x, y = fiber_coords[ifiber,:]
        patch = functools.partial(patches.Circle,(x,y),fibersize,lw=0.25)
        axs['map'].add_patch(patch(fc=bincolors[bin_id],alpha=0.8,ec='none'))
        axs['fmap'].add_patch(patch(fc=fiberfluxcolors['c'][ifiber]))
        axs['smap'].add_patch(patch(fc=fibers2ncolors['c'][ifiber]))
        for k in ['fmap','fmap2','smap','smap2']:
            axs[k].text(x,y,str(fiber_id),**txtkw)
        if fiber_id in goodfibers:
            axs['fmap2'].add_patch(patch(fc=fiberfluxcolors2['c'][i2]))
            axs['smap2'].add_patch(patch(fc=fibers2ncolors2['c'][i2]))
            i2 += 1
    # loop over bins
    for ibin,bin_id in enumerate(bindata['binid']):
        bincolor = bincolors[int(bin_id)]
        x, y = bindata['rx'][ibin], bindata['ry'][ibin]
        # draw bin number at bin center
        axs['map'].plot(x,y,ls='',marker='o',mew=1.0,ms=8.0,mec='k',
                        mfc=bincolor)
        axs['map'].text(x-0.2,y-0.1,str(bin_id),**txtkw)
        # draw polar bin centers
        axs['cent'].plot(x,y,ls='',marker='o',mew=0,ms=5.0,mfc='k')
        if not np.isnan(bindata['rmin'][ibin]):
            pbox = polar_box(bindata['rmin'][ibin],bindata['rmax'][ibin],
                             bindata['thmin'][ibin],bindata['thmax'][ibin])
            patch = functools.partial(descartes.PolygonPatch,pbox,lw=1.5)
            axs['map'].add_patch(patch(fc=bincolor,alpha=0.5,zorder=-1))
            axs['map'].add_patch(patch(fc='none'))
            axs['flux'].add_patch(patch(fc=binfluxcolors['c'][ibin]))
            axs['s2n'].add_patch(patch(fc=bins2ncolors['c'][ibin]))
            for k in ['map','cent','fmap','fmap2','smap','smap2']:
                axs[k].add_patch(patch(fc='none'))
        else:
            patch = functools.partial(patches.Circle,(bindata['x'][ibin],
                                        bindata['y'][ibin]),fibersize,lw=0.25)
            axs['flux'].add_patch(patch(fc=binfluxcolors['c'][ibin]))
            axs['s2n'].add_patch(patch(fc=bins2ncolors['c'][ibin]))
    # draw ma, set axis bounds, save and close
    for k in fig_keys:
        axs[k].add_patch(patches.Circle((0,0),binmeta['r best fullbin'],
                                        ls='dashed',fc='none'))
        axs[k].plot([-binmeta['ma_x'],binmeta['ma_x']],
                    [-binmeta['ma_y'],binmeta['ma_y']],
                    linewidth=1.5,color='r')
        axs[k].axis([-squaremax,squaremax,-squaremax,squaremax])
        pdf.savefig(figs[k])
        plt.close(figs[k])


    # plot flux and s2n vs radius
    t1f = '{} Fiber flux vs radius (with cropping)'.format(gal_name)
    t2f = ('{} Fiber flux vs radius (with cropping and fiber removal)'
           ''.format(gal_name))
    t1s = '{} Fiber s2n vs radius (with cropping)'.format(gal_name)
    t2s = ('{} Fiber s2n vs radius (with cropping and fiber removal)'
           ''.format(gal_name))
    fig1f,ax1f = mplt.scalarmap(figtitle=t1f,xlabel=label_r,ylabel=label_flux)
    fig2f,ax2f = mplt.scalarmap(figtitle=t2f,xlabel=label_r,ylabel=label_flux)
    fig1s,ax1s = mplt.scalarmap(figtitle=t1s,xlabel=label_r,ylabel=label_s2n)
    fig2s,ax2s = mplt.scalarmap(figtitle=t2s,xlabel=label_r,ylabel=label_s2n)
    txtkw = {'horizontalalignment':'center','verticalalignment':'center',
             'fontsize':5}
    for ifiber, (fiber_id,bin_id) in enumerate(zip(fiberids,binids)):
        r = rcoords[ifiber]
        f, s = fiberfluxcolors['x'][ifiber], fibers2ncolors['x'][ifiber]
        if fiber_id in goodfibers:
            ax1f.text(r,f,str(fiber_id),alpha=0.3,**txtkw)
            ax2f.text(r,f,str(fiber_id),**txtkw)
            ax1s.text(r,s,str(fiber_id),alpha=0.3,**txtkw)
            ax2s.text(r,s,str(fiber_id),**txtkw)
        else:
            ax1f.text(r,f,str(fiber_id),**txtkw)
            ax1f.plot(r,f,ls='',marker='o',mec='r',mfc='none',ms=10)
            ax1s.text(r,s,str(fiber_id),**txtkw)
            ax1s.plot(r,s,ls='',marker='o',mec='r',mfc='none',ms=10)
    rmin, rmax = min(rcoords), max(rcoords)
    fmin, fmax = fiberfluxcolors['vmin'], fiberfluxcolors['vmax']
    smin, smax = fibers2ncolors['vmin'], fibers2ncolors['vmax']
    ax1f.axis([rmin,rmax,fmin,fmax])
    ax2f.axis([rmin,rmax,fmin,fmax])
    ax1s.axis([rmin,rmax,smin,smax])
    ax2s.axis([rmin,rmax,smin,smax])
    for ax in [ax1f,ax2f,ax1s,ax2s]:
        ax.set_yscale('log')
        ax.axvline(binmeta['rbinmax'],c='g')
        ax.set_title('(max bin radius marked in green)')
    for fig in [fig1f,fig2f,fig1s,fig2s]:
        pdf.savefig(fig)
        plt.close(fig)

    # plot fiber flux vs noise
    title = '{} All fibers total noise vs flux'.format(gal_name)
    fig,ax = mplt.scalarmap(figtitle=title,xlabel=label_flux,ylabel='noise')
    fibernoise = ifuset.spectrumset.compute_noiseflux()
    for ifiber, fiberid in enumerate(fiberids):
        f, n = fiberfluxcolors['x'][ifiber], fibernoise[ifiber]
        if fiberid in goodfibers:
            ax.text(f,n,str(fiberid),alpha=0.3,**txtkw)
        else:
            ax.text(f,n,str(fiberid),**txtkw)
            ax.plot(f,n,ls='',marker='o',mec='r',mfc='none',ms=10)
    binnoise = specset.compute_noiseflux()
    for ibin,bin_id in enumerate(bindata['binid']):
        f, n = bindata['flux'][ibin], binnoise[ibin]
        n2 = f/bins2ncolors['x'][ibin]
        ax.text(f,n,bin_id,color='g',zorder=-1,**txtkw)
        ax.text(f,n2,bin_id,color='b',zorder=-1,**txtkw)
    bintext1 = 'bins (total flux, total noise)'
    bintext2 = 'bins (total flux, total flux/mean s2n)'
    bintext3 = '(bin values scaled by area)'
    ax.text(0.05,0.95,bintext1,transform=ax.transAxes,color='g')
    ax.text(0.05,0.90,bintext2,transform=ax.transAxes,color='b')
    ax.text(0.05,0.85,bintext3,transform=ax.transAxes,color='k')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fmin = min([min(fiberfluxcolors['x']),min(bindata['flux'])])
    fmax = max([max(fiberfluxcolors['x']),max(bindata['flux'])])
    nmin = min([min(fibernoise),min(binnoise)])
    nmax = max([max(fibernoise),max(binnoise)])
    fakeflux = np.linspace(fmin,fmax,num=100)
    fakenoise = const.flat_plus_poisson(fakeflux,
                                fibermeta['flatnoise'],fibermeta['fluxscale'])
    ax.plot(fakeflux,fakenoise,c='m')
    ax.plot(fakeflux,fakeflux/binmeta['threshold s2n'],c='r')
    ax.axis([fmin,fmax,nmin,nmax])
    pdf.savefig(fig)
    plt.close(fig)


    # plot all fiber spectra
    t1 = '{} All fiber spectra (with cropping)'.format(gal_name)
    t2 = ('{} All fiber spectra (with cropping and fiber removal)'
          ''.format(gal_name))
    fig1,ax1 = mplt.scalarmap(figtitle=t1,xlabel=label_waves,ylabel=label_flux)
    fig2,ax2 = mplt.scalarmap(figtitle=t2,xlabel=label_waves,ylabel=label_flux)
    for ifiber,fiber_id in enumerate(plotfibers):
        fspec = fiberspectra[ifiber]
        ax1.semilogy(fiberwaves,np.abs(fspec),c='r',alpha=0.3)
        ax1.semilogy(fiberwaves,fspec,c='c',alpha=0.3,nonposy='mask') 
        if fiber_id in goodfibers:
            ax2.semilogy(fiberwaves,np.abs(fspec),c='r',alpha=0.3)
            ax2.semilogy(fiberwaves,fspec,c='c',alpha=0.3,nonposy='mask')
    for ax in [ax1,ax2]:
        ax.set_rasterized(True)
        ax.set_title('(red indicates negative values)')
        ax.set_xlim(xmin=fiberwaves[0]-0.05*(fiberwaves[-1]-fiberwaves[0]),
                    xmax=fiberwaves[-1]+0.05*(fiberwaves[-1]-fiberwaves[0]))
    for fig in [fig1,fig2]:
        pdf.savefig(fig)
        plt.close(fig)



    # plot ir for each bin
    fig, ax = mplt.scalarmap(figtitle='{} IR for each bin'.format(gal_name),
                             xlabel=label_waves, ylabel=label_ir,
                             axC_mappable=ircolors['mappable'],
                             axC_label='bin number')
    for ibin in range(nbins):
        ax.plot(specset.waves,specset.metaspectra['ir'][ibin,:],
                c=ircolors['c'][ibin],alpha=0.7)
    pdf.savefig(fig)
    plt.close(fig)



    # plot each spectrum, y-axis also represents bin number
    fig = plt.figure(figsize=(6,nbins))
    fig.suptitle('{} bin spectra by bin number'.format(gal_name))
    yspace = 1/float(nbins)
    ax = fig.add_axes([0.05,0.5*yspace,0.9,1-1.5*yspace])
    for ibin in range(nbins):
        norm = (specset.waves[-1] - specset.waves[0])/bindata['flux'][ibin]
        spectrum = specset.spectra[ibin,:]*norm
        ax.plot(specset.waves,1+ibin-spectrum+spectrum[0],c='k')
    # assuming all 3 full spectra are present, overplot with different colors
    # if the spectra are identical, you will see only the black one
    fullcolors = {0: '0.5', -1: 'k', -2: 'r'}
    full_labels = {0: 'all good fibers', -1: 'all binned fibers',
            -2: 'binned fibers within r={}'.format(binmeta['r best fullbin'])}
    fullnorms = (specset_full.waves[-1] - specset_full.waves[0])
    fullnorms = fullnorms/specset_full.compute_flux()
    for ifull,fullid in reversed(list(enumerate(specset_full.ids))):
        spectrum = specset_full.spectra[ifull,:]*fullnorms[ifull]
        ax.plot(specset_full.waves,-spectrum+spectrum[0],c=fullcolors[fullid],
                label=full_labels[fullid])
    legend = ax.legend(loc=(0,1-1.3/(nbins+3.0)),fontsize=8,
                       title='For full galaxy (bin 0):')
    plt.setp(legend.get_title(),fontsize=8)
    # show where prominent emission lines go
    elines = const.emission_lines
    for eline in elines:
        ax.axvline(elines[eline]['wave'],c='b')
        ax.text(elines[eline]['x'],-1.8+0.3*elines[eline]['y'],
                elines[eline]['name'],fontsize=7,weight='semibold')
    ax.set_xlabel('wavelength ({})'.format(specset.wave_unit))
    ax.set_ylabel('bin number')
    ax.autoscale(tight=True)
    ax.set_ylim(ymin=-2,ymax=nbins+1)
    ax.set_yticks(range(nbins+1))
    ax.set_yticklabels([''] + list(specset.ids))
    ax.yaxis.set_tick_params(color=[0,0,0,0])
    ax.invert_yaxis()
    ax.tick_params(labeltop='on',top='on')
    pdf.savefig(fig)
    plt.close(fig)

    pdf.close()
    return
