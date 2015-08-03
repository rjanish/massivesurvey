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
import massivepy.plot_massive as mplt
from plotting.geo_utils import polar_box


def plot_s2_bin_mitchell(gal_name=None,plot_path=None,raw_cube_path=None,
                         targets_path=None,ir_path=None,fiberinfo_path=None,
                         bininfo_path=None,binspectra_path=None,
                         crop_region=None,fullbin_path=None):
    # read fiber and bin info
    fiberids, binids = np.genfromtxt(fiberinfo_path,
                                     dtype=int,unpack=True)
    bininfo = np.genfromtxt(bininfo_path,names=True,skip_header=1)
    bininfo['thmin'] = 90 + bininfo['thmin']
    bininfo['thmax'] = 90 + bininfo['thmax']
    binsettings = open(bininfo_path,'r').readlines()[18]
    aspect_ratio, s2n_threshold = binsettings.strip().split()[1:]
    aspect_ratio, s2n_threshold = eval(aspect_ratio[:-1]), eval(s2n_threshold)
    nbins = len(bininfo)
    # these lines are a terrible idea.
    ma_line = open(bininfo_path,'r').readlines()[9]
    ma_theta = np.pi/2 + np.deg2rad(float(ma_line.strip().split()[-1]))
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
        if not binid==-99:
            goodfibers.append(fiberid)

    # read spectra fits files
    # (can probably ditch reading the ir, won't actually use it)
    ifuset = ifu.read_raw_datacube(raw_cube_path,targets_path,gal_name,
                                   ir_path=ir_path)
    ifuset.crop(crop_region) # everything plotted here is the cropped version!
    ifuset2 = ifuset.get_subset(goodfibers)
    specset = spec.read_datacube(binspectra_path)
    specset_full = spec.read_datacube(fullbin_path)
    label_waves = r'wavelength (angstroms assumed)'

    # set up fiber coordinate information
    if not all(fiberids==ifuset.spectrumset.ids):
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
    binfluxcolors = mplt.colormap_setup(bininfo['flux'],cmap='Reds')
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
    fiberspectra = ifuset.spectrumset.spectra

    ### plotting begins ###
    pdf = PdfPages(plot_path)

    # plot bin maps, bin fluxes/s2n, bin centers comparison
    # (everything where looping over bins and/or fibers is necessary)
    fig_keys = ['map','flux','s2n','cent','fmap','fmap2','smap','smap2',
                'fvr','fvr2','svr','svr2','spec','spec2']
    titles = ['Bin map (s2n {}, ar {})'.format(s2n_threshold,aspect_ratio),
              'Bin flux map',
              'Bin s2n map',
              'Bin centers (cartesian v polar)',
              'Fiber flux map (with cropping)',
              'Fiber flux map (with cropping and fiber removal)',
              'Fiber s2n map (with cropping)',
              'Fiber s2n map (with cropping and fiber removal)',
              'Fiber flux vs radius (with cropping)',
              'Fiber flux vs radius (with cropping and fiber removal)',
              'Fiber s2n vs radius (with cropping)',
              'Fiber s2n vs radius (with cropping and fiber removal)',
              'All fiber spectra (with cropping)',
              'All fiber spectra (with cropping and fiber removal)']
    xlabels = 8*[label_x] + 4*[label_r] + 2*[label_waves]
    ylabels = 8*[label_y] + 2*[label_flux] + 2*[label_s2n] + 2*[label_flux]
    mappables = [None,binfluxcolors['mappable'],bins2ncolors['mappable'],None,
                 fiberfluxcolors['mappable'],fiberfluxcolors2['mappable'],
                 fibers2ncolors['mappable'],fibers2ncolors2['mappable'],
                 None,None,None,None,None,None]
    mlabels = [None,label_flux,label_s2n,None,label_flux,label_flux,
               label_s2n,label_s2n,None,None,None,None,None,None]
    figs,axs = {}, {}
    # create figures and axes
    for i,k in enumerate(fig_keys):
        figs[k],axs[k] = mplt.scalarmap(figtitle=titles[i],
                                        xlabel=xlabels[i],ylabel=ylabels[i],
                                        axC_mappable=mappables[i],
                                        axC_label=mlabels[i])
    # draw cartesian bin centers
    axs['cent'].plot(bininfo['x'],bininfo['y'],ls='',marker='s',mew=0,ms=5.0,
                     mfc='r')
    # loop over fibers
    txtkw = {'horizontalalignment':'center','verticalalignment':'center',
             'fontsize':5}
    i2 = 0 # a hacky way to access only the fluxes for good fibers
    for fiber_id,bin_id in zip(fiberids,binids):
        x, y = fiber_coords[fiber_id,:]
        r = rcoords[fiber_id]
        f, s = fiberfluxcolors['x'][fiber_id], fibers2ncolors['x'][fiber_id]
        patch = functools.partial(patches.Circle,(x,y),fibersize,lw=0.25)
        axs['map'].add_patch(patch(fc=bincolors[bin_id],alpha=0.8,ec='none'))
        axs['fmap'].add_patch(patch(fc=fiberfluxcolors['c'][fiber_id]))
        axs['smap'].add_patch(patch(fc=fibers2ncolors['c'][fiber_id]))
        axs['fmap'].text(x,y,str(fiber_id),**txtkw)
        axs['fmap2'].text(x,y,str(fiber_id),**txtkw)
        axs['smap'].text(x,y,str(fiber_id),**txtkw)
        axs['smap2'].text(x,y,str(fiber_id),**txtkw)
        if fiber_id in plotfibers:
            axs['spec'].semilogy(fiberwaves,np.abs(fiberspectra[fiber_id]),
                                 c='r',alpha=0.3)
            axs['spec'].semilogy(fiberwaves,fiberspectra[fiber_id],
                                 c='c',alpha=0.3,nonposy='mask') 
        if fiber_id in goodfibers:
            axs['fmap2'].add_patch(patch(fc=fiberfluxcolors2['c'][i2]))
            axs['smap2'].add_patch(patch(fc=fibers2ncolors2['c'][i2]))
            i2 += 1
            axs['fvr'].text(r,f,str(fiber_id),alpha=0.3,**txtkw)
            axs['fvr2'].text(r,f,str(fiber_id),**txtkw)
            axs['svr'].text(r,s,str(fiber_id),alpha=0.3,**txtkw)
            axs['svr2'].text(r,s,str(fiber_id),**txtkw)
            if fiber_id in plotfibers:
                axs['spec2'].semilogy(fiberwaves,np.abs(fiberspectra[fiber_id]),
                                      c='r',alpha=0.3)
                axs['spec2'].semilogy(fiberwaves,fiberspectra[fiber_id],
                                      c='c',alpha=0.3,nonposy='mask')
        else:
            axs['fvr'].text(r,f,str(fiber_id),**txtkw)
            axs['fvr'].plot(r,f,ls='',marker='o',mec='r',mfc='none',ms=10)
            axs['svr'].text(r,s,str(fiber_id),**txtkw)
            axs['svr'].plot(r,s,ls='',marker='o',mec='r',mfc='none',ms=10)
    # loop over bins
    for ibin,bin_id in enumerate(bininfo['binid']):
        bincolor = bincolors[int(bin_id)]
        # draw bin number at bin center
        xbin = -bininfo['r'][ibin]*np.sin(np.deg2rad(bininfo['th'][ibin]))
        ybin = bininfo['r'][ibin]*np.cos(np.deg2rad(bininfo['th'][ibin]))
        axs['map'].plot(xbin,ybin,ls='',marker='o',mew=1.0,ms=8.0,mec='k',
                        mfc=bincolor)
        axs['map'].text(xbin-0.2,ybin-0.1,str(int(bin_id)),fontsize=5,
                        horizontalalignment='center',verticalalignment='center')
        # draw polar bin centers
        axs['cent'].plot(xbin,ybin,ls='',marker='o',mew=0,ms=5.0,mfc='k')
        if not np.isnan(bininfo['rmin'][ibin]):
            pbox = polar_box(bininfo['rmin'][ibin],bininfo['rmax'][ibin],
                             bininfo['thmin'][ibin],bininfo['thmax'][ibin])
            patch = functools.partial(descartes.PolygonPatch,pbox,lw=1.5)
            axs['map'].add_patch(patch(fc=bincolor,alpha=0.5,zorder=-1))
            axs['map'].add_patch(patch(fc='none'))
            axs['flux'].add_patch(patch(fc=binfluxcolors['c'][ibin]))
            axs['s2n'].add_patch(patch(fc=bins2ncolors['c'][ibin]))
            axs['cent'].add_patch(patch(fc='none'))
            axs['fmap'].add_patch(patch(fc='none'))
            axs['fmap2'].add_patch(patch(fc='none'))
            axs['smap'].add_patch(patch(fc='none'))
            axs['smap2'].add_patch(patch(fc='none'))
        else:
            patch = functools.partial(patches.Circle,(bininfo['x'][ibin],
                                        bininfo['y'][ibin]),fibersize,lw=0.25)
            axs['flux'].add_patch(patch(fc=binfluxcolors['c'][ibin]))
            axs['s2n'].add_patch(patch(fc=bins2ncolors['c'][ibin]))
    # draw ma, set axis bounds
    rmax = np.nanmax(bininfo['rmax'])
    for k in ['map','flux','s2n','cent','fmap','fmap2','smap','smap2']:
        axs[k].plot([-rmax*1.1*np.cos(ma_theta), rmax*1.1*np.cos(ma_theta)],
                    [-rmax*1.1*np.sin(ma_theta), rmax*1.1*np.sin(ma_theta)],
                    linewidth=1.5, color='r')
        axs[k].axis([-squaremax,squaremax,-squaremax,squaremax])
    for k in['fvr','fvr2']:
        axs[k].set_yscale('log')
        axs[k].axis([min(rcoords),max(rcoords),
                     fiberfluxcolors['vmin'],fiberfluxcolors['vmax']])
    for k in['svr','svr2']:
        axs[k].set_yscale('log')
        axs[k].axis([min(rcoords),max(rcoords),
                     fibers2ncolors['vmin'],fibers2ncolors['vmax']])
    for k in ['spec','spec2']:
        axs[k].set_rasterized(True)
        axs[k].set_title('(red indicates negative values)')
        axs[k].set_xlim(xmin=fiberwaves[0]-0.05*(fiberwaves[-1]-fiberwaves[0]),
                        xmax=fiberwaves[-1]+0.05*(fiberwaves[-1]-fiberwaves[0]))
    # save and close
    for k in fig_keys:
        pdf.savefig(figs[k])
        plt.close(figs[k])


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

    pdf.close()
    return
