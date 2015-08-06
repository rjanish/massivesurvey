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
import utilities as utl
from plotting.geo_utils import polar_box


def plot_s3_fullfit(gal_name=None,plot_path=None,templates_dir=None,
                    binspectra_path=None,main_output=None,
                    temps_output=None,fit_range=None,mask=None):
    # get data from fits files of ppxf fit output
    fitdata = mpio.get_friendly_ppxf_output(main_output)
    nbins = fitdata['nbins']
    nmoments = fitdata['nmoments']
    moment_names = ['h{}'.format(m+1) for m in range(nmoments)]
    moment_names[0] = 'V'
    moment_names[1] = 'sigma'
    # get spectrum and bin information
    specset = spec.read_datacube(binspectra_path)
    specset = specset.get_subset(fitdata['bins']['id'])
    # save "friendly" text output for theorists
    txtfile_header = 'Columns are as follows:'
    colnames = fitdata['temps'].dtype.names
    txtfile_header += '\n ' + ' '.join(colnames)
    txtfile_header += '\n{} nonzero templates out of {}'.format(
        len(fitdata['temps']),fitdata['ntemps'])
    fmt = ['%i']
    fmt.extend(['%-8g']*(len(colnames)-1))
    np.savetxt(temps_output,fitdata['temps'],fmt=fmt,
               header=txtfile_header,delimiter='\t')


    ### Plotting Begins! ###

    pdf = PdfPages(plot_path)

    # template plots
    catalogfile = os.path.join(templates_dir,'catalog.txt')
    catalog = pd.read_csv(catalogfile,index_col='miles_id')
    spectype = np.array(catalog['spt'][fitdata['temps']['id']],dtype='S1')
    # sort by spectype for pie chart
    ii = np.argsort(spectype,kind='mergesort')
    templates = fitdata['temps']['id'][ii]
    weights = fitdata['temps']['weight'][ii]
    fluxweights = fitdata['temps']['fluxweight'][ii]
    spectype = spectype[ii]
    spt_long = catalog['spt'][templates]
    pielabels = ["{} ({})".format(s,t) for s,t in zip(spt_long,templates)]
    piecolors = [const.spectype_colors[s[0]] for s in spectype]
    # plot raw weights
    fig = plt.figure(figsize=(6,5))
    fig.suptitle('Templates (raw weights)')
    ax = fig.add_axes([0.17,0.05,0.7,0.7*1.2])
    patches, labels, txt = ax.pie(weights,labels=pielabels,
                                  colors=piecolors,labeldistance=1.1,
                                  autopct='%1.1f%%',wedgeprops={'lw':0.2})
    for label in labels: label.set_fontsize(7)
    pdf.savefig(fig)
    plt.close(fig)
    # plot flux-normalized weights
    fig = plt.figure(figsize=(6,5))
    fig.suptitle('Templates (flux-normalized weights)')
    ax = fig.add_axes([0.15,0.05,0.7,0.7*1.2])
    patches, labels, txt = ax.pie(fluxweights,labels=pielabels,
                                  colors=piecolors,labeldistance=1.1,
                                  autopct='%1.1f%%',wedgeprops={'lw':0.2})
    for label in labels: label.set_fontsize(7)
    pdf.savefig(fig)
    plt.close(fig)
    
    # plot the fit spectrum
    fig = plt.figure(figsize=(6, 5))
    fig.suptitle('full galaxy spectrum fit')
    ax = fig.add_axes([0.15,0.1,0.7,0.7])
    target_specset = specset.crop(fit_range)
    spectrum = target_specset.spectra[0]
    spectrum = spectrum/np.median(spectrum)
    waves = target_specset.waves
    model = fitdata['spec']['bestmodel'][0]
    modelwaves = fitdata['waves']
    # modelwaves should be same as waves, but is longer by one pixel!!
    # this bug shows up only (so far) in NGC1129
    ax.plot(waves,spectrum-spectrum[0],c='k')
    ax.plot(modelwaves,model-spectrum[0],c='r',lw=0.7)
    ax.text(waves[0],0.4,
            r'$\chi^2={:4.2f}$'.format(fitdata['bins']['chisq'][0]))
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
        ax.text(elines[eline]['x'],-0.05*elines[eline]['y'],
                elines[eline]['name'],fontsize=7,weight='semibold')
    ax.set_xlabel('wavelength ({})'.format("units"))
    ax.set_ylabel('bin number')
    ax.set_yticks([0])
    ax.autoscale(tight=True)
    pdf.savefig(fig)
    plt.close(fig)
    
    pdf.close()
    return

def plot_s3_binfit(gal_name=None,plot_path=None,binspectra_path=None,
                   bininfo_path=None,main_output=None,mc_output=None,
                   moments_output=None,mcmoments_output=None,fit_range=None,
                   mask=None,compare_moments=None,compare_bins=None):
    # get data from fits files of ppxf fit output
    fitdata = mpio.get_friendly_ppxf_output(main_output)
    nbins = fitdata['nbins']
    nmoments = fitdata['nmoments']
    moment_names = ['h{}'.format(m+1) for m in range(nmoments)]
    moment_names[0] = 'V'
    moment_names[1] = 'sigma'
    # get spectrum and bin information
    specset = spec.read_datacube(binspectra_path)
    specset = specset.get_subset(fitdata['bins']['id'])
    bininfo = np.genfromtxt(bininfo_path,names=True,skip_header=1)
    bininfo['thmin'] = 90 + bininfo['thmin']
    bininfo['thmax'] = 90 + bininfo['thmax']
    squaremax = np.nanmax(bininfo['rmax'])
    coordunit = 'arcsec'
    label_x = r'$\leftarrow$east ({}) west$\rightarrow$'.format(coordunit)
    label_y = r'$\leftarrow$south ({}) north$\rightarrow$'.format(coordunit)
    ibins_all = {int(bininfo['binid'][i]):i for i in range(len(bininfo))}
    ibins = [ibins_all[binid] for binid in fitdata['bins']['id']]
    if os.path.isfile(mc_output):
        have_mc = True
        mcdata = mpio.get_friendly_ppxf_output_mc(plot_info['mc_output'])
    else:
        have_mc = False

    # save "friendly" text output for theorists
    txtfile_array = np.zeros((nbins,1+2*nmoments))
    txtfile_header = 'Fit results for {}'.format(gal_name)
    txtfile_header += '\nPPXF input parameters were as follows:'
    for param in ['add_deg', 'mul_deg']:
        txtfile_header += '\n {} = {}'.format(param,fitdata[param])
    txtfile_array[:,0] = fitdata['bins']['id']
    txtfile_array[:,1:1+nmoments] = fitdata['gh']['moment']
    if have_mc:
        txtfile_array[:,-nmoments:] = mcdata['err']
        txtfile_header += '\nErrors from {} mc runs'.format(mcdata['nruns'])
    else:
        txtfile_array[:,-nmoments:] = fitdata['gh']['scalederr']
        txtfile_header += '\nErrors from ppxf, scaled'
    txtfile_header += '\nColumns are as follows:'
    colnames = ['bin'] + moment_names + [m+'err' for m in moment_names]
    txtfile_header += '\n' + ' '.join(colnames)
    fmt = ['%i'] + 2*nmoments*['%-6f']
    np.savetxt(moments_output,txtfile_array,fmt=fmt,
               delimiter='\t',header=txtfile_header)

    # check for mc runs
    if have_mc:
        if os.path.isdir(plot_info['mcmoments_output']):
            shutil.rmtree(plot_info['mcmoments_output'])
        os.mkdir(plot_info['mcmoments_output'])
        txtfile_header = 'Columns are as follows:'
        txtfile_header += '\n' + ' '.join(moment_names)
        fmt = nmoments*['%-6f']
        for ibin,binid in enumerate(fitdata['bins']['id']):
            binpath = os.path.join(plot_info['mcmoments_output'],
                                   'bin{:d}.txt'.format(binid))
            np.savetxt(binpath,mcdata['moments'][ibin].T,fmt=fmt,
                       delimiter='\t',header=txtfile_header)

    # prep comparison plot info, if available
    if not compare_moments=='none':
        do_comparison = True
    else:
        do_comparison = False
    if do_comparison:
        fitdata2 = mpio.get_friendly_ppxf_output(plot_info['compare_moments'])
        bininfo2 = np.genfromtxt(plot_info['compare_bins'],names=True,
                                 skip_header=12)
        ibins_all2 = {int(bininfo2['binid'][i]):i for i in range(len(bininfo2))}
        ibins2 = [ibins_all2[binid] for binid in fitdata2['bins']['id']]

    ### Plotting Begins! ###

    pdf = PdfPages(plot_path)
    # moments plots
    for i in range(nmoments):
        fig = plt.figure(figsize=(6,6))
        fig.suptitle('Moment vs radius ({})'.format(moment_names[i]))
        ax = fig.add_axes([0.15,0.1,0.8,0.7])
        moments = fitdata['gh']['moment'][:,i]
        moments_r = bininfo['r'][ibins]
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
            ax.plot(bininfo2['r'][ibins2],fitdata2['gh']['moment'][:,i],
                    ls='',marker='s',mfc='g',ms=5.0,alpha=0.8,
                    label='comparison run')
            mainlabel = 'this run'
        # plot moments
        ax.plot(moments_r,moments,ls='',marker='o',mfc='b',ms=5.0,alpha=0.8,
                label=mainlabel)
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
    V0 = [np.average(Vdata,weights=bininfo['flux']),np.average(Vdata),
          utl.median(Vdata,weights=bininfo['flux']),np.median(Vdata)]
    Vtitles = ['2D map of centered V (flux weighted average, V0={:.2f})'.format,
               '2D map of centered V (average, V0={:.2f})'.format,
               '2D map of centered V (flux weighted median, V0={:.2f})'.format,
               '2D map of centered V (median, V0={:.2f})'.format]
    for i in range(4):
        Vcolors = mplt.lin_colormap_setup(Vdata-V0[i],cmap='BrBG',center=True)
        fig, ax = mplt.scalarmap(figtitle=Vtitles[i](V0[i]),
                                 xlabel=label_x,ylabel=label_y,
                                 axC_mappable=Vcolors['mappable'],axC_label='V')
        for ibin in range(nbins):
            xbin = -bininfo['r'][ibin]*np.sin(np.deg2rad(bininfo['th'][ibin]))
            ybin = bininfo['r'][ibin]*np.cos(np.deg2rad(bininfo['th'][ibin]))
            if not np.isnan(bininfo['rmin'][ibin]):
                pbox = polar_box(bininfo['rmin'][ibin],bininfo['rmax'][ibin],
                                 bininfo['thmin'][ibin],bininfo['thmax'][ibin])
                patch = functools.partial(descartes.PolygonPatch,pbox,lw=1.5)
                ax.add_patch(patch(fc=Vcolors['c'][ibin],zorder=-1))
            else:
                patch = functools.partial(patches.Circle,(bininfo['x'][ibin],
                                        bininfo['y'][ibin]),fibersize,lw=0.25)
                ax.add_patch(patch(fc=Vcolors['c'][ibin]))
        ax.axis([-squaremax,squaremax,-squaremax,squaremax])
        pdf.savefig(fig)
        plt.close(fig)        

    # do all the moments
    momentcmaps = ['Blues','Purples'] + (nmoments-2)*['bwr']
    center = 2*[False] + (nmoments-2)*[True]
    for i in range(nmoments):
        momentdata = fitdata['gh']['moment'][:,i]
        momentcolors = mplt.lin_colormap_setup(momentdata,cmap=momentcmaps[i],
                                               center=center[i])
        title = '2D map of {}'.format(moment_names[i])
        fig, ax = mplt.scalarmap(figtitle=title,xlabel=label_x,ylabel=label_y,
                                 axC_mappable=momentcolors['mappable'],
                                 axC_label=moment_names[i])
        for ibin in range(nbins):
            xbin = -bininfo['r'][ibin]*np.sin(np.deg2rad(bininfo['th'][ibin]))
            ybin = bininfo['r'][ibin]*np.cos(np.deg2rad(bininfo['th'][ibin]))
            if not np.isnan(bininfo['rmin'][ibin]):
                pbox = polar_box(bininfo['rmin'][ibin],bininfo['rmax'][ibin],
                                 bininfo['thmin'][ibin],bininfo['thmax'][ibin])
                patch = functools.partial(descartes.PolygonPatch,pbox,lw=1.5)
                ax.add_patch(patch(fc=momentcolors['c'][ibin],zorder=-1))
            else:
                patch = functools.partial(patches.Circle,(bininfo['x'][ibin],
                                        bininfo['y'][ibin]),fibersize,lw=0.25)
                ax.add_patch(patch(fc=momentcolors['c'][ibin]))
        ax.axis([-squaremax,squaremax,-squaremax,squaremax])
        pdf.savefig(fig)
        plt.close(fig)

    # plot each spectrum, y-axis also represents bin number
    #figheight = max(fitdata['bins']['id'])
    #figheight = max(figheight,4)
    fig = plt.figure(figsize=(6, nbins+3))
    fig.suptitle('bin spectra by bin number')
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
        ax.text(elines[eline]['x'],-0.9+0.15*elines[eline]['y'],
                elines[eline]['name'],fontsize=7,weight='semibold')
    ax.set_xlabel('wavelength ({})'.format("units"))
    ax.set_ylabel('bin number')
    ax.autoscale(tight=True)
    ax.set_yticks(range(nbins))
    ax.set_yticklabels(fitdata['bins']['id'])
    ax.set_ylim(ymin=-1,ymax=nbins)
    ax.invert_yaxis()
    ax.tick_params(labeltop='on',top='on')
    pdf.savefig(fig)
    plt.close(fig)
    
    pdf.close()
    return
