"""
MASSIVE-specific plotting routines:

This file contains the main plotting fuctions for s3_ppxf_fitspectra.
"""

import os
import shutil
import functools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
import descartes

import massivepy.constants as const
import massivepy.spectrum as spec
import massivepy.io as mpio
import massivepy.plot_massive as mplt
import massivepy.binning as binning
import utilities as utl
from plotting.geo_utils import polar_box


def plot_s3_fullfit(gal_name=None,plot_path=None,templates_dir=None,
                    binspectra_path=None,main_output=None,
                    temps_output=None,fit_range=None,mask=None):
    # get data from fits files of ppxf fit output
    fitdata = mpio.get_friendly_ppxf_output(main_output)
    nbins = fitdata['metadata']['nbins']
    nmoments = fitdata['metadata']['nmoments']
    moment_names = ['h{}'.format(m+1) for m in range(nmoments)]
    moment_names[0] = 'V'
    moment_names[1] = 'sigma'
    # get spectrum and bin information
    specset = spec.read_datacube(binspectra_path)
    specset = specset.get_subset(fitdata['bins']['id'])

    ### Plotting Begins! ###

    pdf = PdfPages(plot_path)

    # template plots
    catalogfile = os.path.join(templates_dir,'catalog.txt')
    catalog = pd.read_csv(catalogfile,index_col='miles_id')
    for i in range(nbins):
        binid = fitdata['bins']['id'][i]
        temps = fitdata['temps'][i,:]
        ii = np.nonzero(temps['weight']) # use only nonzero templates
        temps = temps[ii]
        spectype = np.array(catalog['spt'][temps['id']])
        ii = np.argsort([s[0] for s in spectype],kind='mergesort')
        temps, spectype = temps[ii], spectype[ii]
        pielabels = ["{} ({})".format(s,t) for s,t in zip(spectype,temps['id'])]
        piecolors = [const.spectype_colors[s[0]] for s in spectype]
        # plot raw weights
        fig = plt.figure(figsize=(6,5))
        fig.suptitle('{} Templates (raw weights) bin {}'.format(gal_name,binid))
        ax = fig.add_axes([0.17,0.05,0.7,0.7*1.2])
        piepatches, labels, txt = ax.pie(temps['weight'],labels=pielabels,
                                         colors=piecolors,labeldistance=1.1,
                                        autopct='%1.1f%%',wedgeprops={'lw':0.2})
        for label in labels: label.set_fontsize(7)
        pdf.savefig(fig)
        plt.close(fig)
        # plot flux-normalized weights
        fig = plt.figure(figsize=(6,5))
        fig.suptitle('{} Templates (flux-normalized weights) bin {}'
                     ''.format(gal_name,binid))
        ax = fig.add_axes([0.15,0.05,0.7,0.7*1.2])
        piepatches, labels, txt = ax.pie(temps['fluxweight'],labels=pielabels,
                                         colors=piecolors,labeldistance=1.1,
                                        autopct='%1.1f%%',wedgeprops={'lw':0.2})
        for label in labels: label.set_fontsize(7)
        pdf.savefig(fig)
        plt.close(fig)
    
    # plot the fit spectrum
    fig = plt.figure(figsize=(6, nbins+3))
    fig.suptitle('{} Full galaxy spectra by bin number'.format(gal_name))
    yspace = 1/float(nbins+3)
    ax = fig.add_axes([0.05,0.5*yspace,0.9,1-1.5*yspace])
    target_specset = specset.crop(fit_range)
    for i,binid in enumerate(fitdata['bins']['id']):
        spectrum = target_specset.spectra[i]
        spectrum = spectrum/np.median(spectrum)
        waves = target_specset.waves
        model = fitdata['spec']['bestmodel'][i]
        modelwaves = fitdata['waves']
        # modelwaves should be same as waves, but is longer by one pixel!!
        # this bug shows up only (so far) in NGC1129
        ax.plot(waves,i-spectrum+spectrum[0],c='k')
        ax.plot(modelwaves,i-model+spectrum[0],c='r',lw=0.7)
        ax.text(waves[0],i-0.4,
                r'$\chi^2={:4.2f}$'.format(fitdata['bins']['chisq'][i]))
    # find regions to mask
    # should add masking of bad_data as well!
    for m in mask:
        ax.axvspan(m[0],m[1],fc='k',ec='none',alpha=0.5,lw=0)
    # mark prominent emission lines
    elines = const.emission_lines
    for eline in elines:
        if elines[eline]['wave']<waves[0] or elines[eline]['wave']>waves[-1]:
            continue
        ax.axvline(elines[eline]['wave'],c='b')
        ax.text(elines[eline]['x'],-1.9+0.1*elines[eline]['y'],
                elines[eline]['name'],fontsize=7,weight='semibold')
    ax.set_xlabel('wavelength ({})'.format("units"))
    ax.set_ylabel('bin number')
    ax.set_yticks([0])
    ax.autoscale(tight=True)
    ax.set_yticks(range(nbins))
    ax.set_yticklabels(fitdata['bins']['id'])
    ax.set_ylim(ymin=-2,ymax=nbins)
    ax.invert_yaxis()
    ax.tick_params(labeltop='on',top='on')
    pdf.savefig(fig)
    plt.close(fig)
    
    pdf.close()
    return

def plot_s3_binfit(gal_name=None,plot_path=None,binspectra_path=None,
                   bininfo_path=None,main_output=None,mc_output=None,
                   moments_output=None,mcmoments_output=None,fit_range=None,
                   mask=None,compare_labels=None,compare_moments=None,
                   compare_bins=None,templates_dir=None):
    # get data from fits files of ppxf fit output
    fitdata = mpio.get_friendly_ppxf_output(main_output)
    nbins = fitdata['metadata']['nbins']
    nmoments = fitdata['metadata']['nmoments']
    moment_names = ['h{}'.format(m+1) for m in range(nmoments)]
    moment_names[0] = 'V'
    moment_names[1] = 'sigma'
    # get spectrum and bin information
    specset = spec.read_datacube(binspectra_path)
    specset = specset.get_subset(fitdata['bins']['id'])
    bindata, binetc = binning.read_bininfo(bininfo_path)
    coordunit = 'arcsec'
    label_x = r'$\leftarrow$east ({}) west$\rightarrow$'.format(coordunit)
    label_y = r'$\leftarrow$south ({}) north$\rightarrow$'.format(coordunit)
    ibins_all = {int(bindata['binid'][i]):i for i in range(len(bindata))}
    ibins = [ibins_all[binid] for binid in fitdata['bins']['id']]
    if os.path.isfile(mc_output):
        have_mc = True
        mcdata = mpio.get_friendly_ppxf_output_mc(mc_output)
    else:
        have_mc = False

    # prep comparison plot info, if available
    if not compare_moments=='none':
        do_comparison = True
    else:
        do_comparison = False
    if do_comparison:
        fitdata2 = mpio.get_friendly_ppxf_output(compare_moments)
        bindata2, binetc2 = binning.read_bininfo(compare_bins)
        ibins_all2 = {int(bindata2['binid'][i]):i for i in range(len(bindata2))}
        ibins2 = [ibins_all2[binid] for binid in fitdata2['bins']['id']]

    ### Plotting Begins! ###

    pdf = PdfPages(plot_path)
    # moments plots
    for i in range(nmoments):
        fig = plt.figure(figsize=(6,6))
        fig.suptitle('{} Moment vs radius ({})'
                     ''.format(gal_name,moment_names[i]))
        ax = fig.add_axes([0.15,0.1,0.8,0.7])
        moments = fitdata['gh']['moment'][:,i]
        moments_r = bindata['r'][ibins]
        moments_err = fitdata['gh']['scalederr'][:,i]
        ax.errorbar(moments_r,moments,yerr=moments_err,ls='',
                    marker=None,ecolor='0.7',elinewidth=0.7,label='ppxf')
        # if available, plot better mc errors and comparison points
        if have_mc:
            mc_err = mcdata['err']
            ax.errorbar(moments_r,moments,yerr=mcdata['err'][:,i],ls='',
                        marker=None,ecolor='k',elinewidth=1.0,label='mc')
        mainlabel = None
        if do_comparison:
            moments2 = fitdata2['gh']['moment'][:,i]
            moments_r2 = bindata2['r'][ibins2]
            ax.plot(moments_r2,moments2,ls='',marker='s',mfc='g',ms=5.0,
                    alpha=0.8,label=compare_labels[0])
            mainlabel = compare_labels[1]
        # plot moments
        ax.plot(moments_r,moments,ls='',marker='o',mfc='c',ms=7.0,alpha=0.8,
                label=mainlabel)
        for imom,ibin in enumerate(ibins):
            ax.text(moments_r[imom]-0.002*max(moments_r),moments[imom],
                    str(int(bindata['binid'][ibin])),fontsize=5,
                    horizontalalignment='center',verticalalignment='center')
        # symmetrize y axis for all but v and sigma
        if not i in (0,1):
            ylim = max(np.abs(ax.get_ylim()))
            ax.set_ylim(ymin=-ylim,ymax=ylim)
        ax.legend(loc='lower center',bbox_to_anchor=(0.5,1),ncol=2)
        ax.set_xlabel('radius')
        ax.set_ylabel(moment_names[i])
        pdf.savefig(fig)
        plt.close(fig)            
    
    # 2D kinematic maps at last, wheee
    fibersize=const.mitchell_fiber_radius.value

    # do 4 versions of centered V
    Vdata = fitdata['gh']['moment'][:,0]
    V0 = [np.average(Vdata,weights=bindata['flux'][ibins]),np.average(Vdata),
          utl.median(Vdata,weights=bindata['flux'][ibins]),np.median(Vdata)]
    Vtitles = ['{} 2D map of centered V (flux weighted avg, V0={:.2f})'.format,
               '{} 2D map of centered V (average, V0={:.2f})'.format,
               '{} 2D map of centered V (flux weighted med., V0={:.2f})'.format,
               '{} 2D map of centered V (median, V0={:.2f})'.format]
    for i in range(4):
        Vcolors = mplt.lin_colormap_setup(Vdata-V0[i],cmap='BrBG',center=True)
        fig, ax = mplt.scalarmap(figtitle=Vtitles[i](gal_name,V0[i]),
                                 xlabel=label_x,ylabel=label_y,
                                 axC_mappable=Vcolors['mappable'],axC_label='V')
        for ibin in range(nbins):
            if not np.isnan(bindata['rmin'][ibin]):
                pbox = polar_box(bindata['rmin'][ibin],bindata['rmax'][ibin],
                                 bindata['thmin'][ibin],bindata['thmax'][ibin])
                patch = functools.partial(descartes.PolygonPatch,pbox,lw=1.5)
                ax.add_patch(patch(fc=Vcolors['c'][ibin],zorder=-1))
            else:
                patch = functools.partial(patches.Circle,(bindata['x'][ibin],
                                        bindata['y'][ibin]),fibersize,lw=0.25)
                ax.add_patch(patch(fc=Vcolors['c'][ibin]))
        ax.axis([-binetc['rbinmax'],binetc['rbinmax'],
                 -binetc['rbinmax'],binetc['rbinmax']])
        pdf.savefig(fig)
        plt.close(fig)        

    # do all the moments
    momentcmaps = ['Blues','Purples'] + (nmoments-2)*['bwr']
    center = 2*[False] + (nmoments-2)*[True]
    for i in range(nmoments):
        momentdata = fitdata['gh']['moment'][:,i]
        momentcolors = mplt.lin_colormap_setup(momentdata,cmap=momentcmaps[i],
                                               center=center[i])
        title = '{} 2D map of {}'.format(gal_name,moment_names[i])
        fig, ax = mplt.scalarmap(figtitle=title,xlabel=label_x,ylabel=label_y,
                                 axC_mappable=momentcolors['mappable'],
                                 axC_label=moment_names[i])
        for ibin in range(nbins):
            if not np.isnan(bindata['rmin'][ibin]):
                pbox = polar_box(bindata['rmin'][ibin],bindata['rmax'][ibin],
                                 bindata['thmin'][ibin],bindata['thmax'][ibin])
                patch = functools.partial(descartes.PolygonPatch,pbox,lw=1.5)
                ax.add_patch(patch(fc=momentcolors['c'][ibin],zorder=-1))
            else:
                patch = functools.partial(patches.Circle,(bindata['x'][ibin],
                                        bindata['y'][ibin]),fibersize,lw=0.25)
                ax.add_patch(patch(fc=momentcolors['c'][ibin]))
        ax.axis([-binetc['rbinmax'],binetc['rbinmax'],
                 -binetc['rbinmax'],binetc['rbinmax']])
        pdf.savefig(fig)
        plt.close(fig)

    # plot each spectrum, y-axis also represents bin number
    fig = plt.figure(figsize=(6, nbins+3))
    fig.suptitle('{} bin spectra by bin number'.format(gal_name))
    yspace = 1/float(nbins+3)
    ax = fig.add_axes([0.05,0.5*yspace,0.9,1-1.5*yspace])
    target_specset = specset.crop(fit_range)
    for i,binid in enumerate(fitdata['bins']['id']):
        spectrum = target_specset.get_subset([binid]).spectra[0]
        spectrum = spectrum/np.median(spectrum)
        waves = target_specset.waves
        model = fitdata['spec']['bestmodel'][i]
        modelwaves = fitdata['waves']
        # modelwaves should be same as waves, but is longer by one pixel!!
        # this bug shows up only (so far) in NGC1129
        ax.plot(waves,i-spectrum+spectrum[0],c='k')
        ax.plot(modelwaves,i-model+spectrum[0],c='r',lw=0.7)
        ax.text(waves[0],binid-0.4,
                r'$\chi^2={:4.2f}$'.format(fitdata['bins']['chisq'][i]))
    # find regions to mask
    # should add masking of bad_data as well!
    for m in mask:
        ax.axvspan(m[0],m[1],fc='k',ec='none',alpha=0.5,lw=0)
    # mark prominent emission lines
    elines = const.emission_lines
    for eline in elines:
        if elines[eline]['wave']<waves[0] or elines[eline]['wave']>waves[-1]:
            continue
        ax.axvline(elines[eline]['wave'],c='b')
        ax.text(elines[eline]['x'],-1.9+0.15*elines[eline]['y'],
                elines[eline]['name'],fontsize=7,weight='semibold')
    ax.set_xlabel('wavelength ({})'.format("units"))
    ax.set_ylabel('bin number')
    ax.autoscale(tight=True)
    ax.set_yticks(range(nbins))
    ax.set_yticklabels(fitdata['bins']['id'])
    ax.set_ylim(ymin=-2,ymax=nbins)
    ax.invert_yaxis()
    ax.tick_params(labeltop='on',top='on')
    pdf.savefig(fig)
    plt.close(fig)
    

    # template plots
    catalogfile = os.path.join(templates_dir,'catalog.txt')
    catalog = pd.read_csv(catalogfile,index_col='miles_id')
    fig_ar = ((nbins-1)/5 + 2)/5.0
    fig = plt.figure(figsize=(6,fig_ar*6))
    fig.suptitle('{} Template weights for each bin (raw weights)'
                 ''.format(gal_name))
    for ibin in range(nbins):
        temps = fitdata['temps'][ibin,:]
        spectype = np.array(catalog['spt'][temps['id']])
        ii = np.argsort([s[0] for s in spectype],kind='mergesort')
        temps, spectype = temps[ii], spectype[ii]
        piecolors = [const.spectype_colors[s[0]] for s in spectype]
        irow, icol = ibin/5, ibin%5
        width, height = 0.2, 0.2/fig_ar
        ax = fig.add_axes([icol*width,1-(irow+2)*height,width,height])
        ax.pie(temps['weight'],colors=piecolors,wedgeprops={'lw':0},radius=1.2)
        ax.plot(0,0,marker='o',mfc='w',ms=12.0)
        ax.text(-0.02,-0.01,fitdata['bins']['id'][ibin],fontsize=8.0,
                horizontalalignment='center',verticalalignment='center')
    pdf.savefig(fig)
    plt.close(fig)

    pdf.close()

    return
