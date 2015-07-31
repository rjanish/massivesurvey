"""
MASSIVE-specific plotting routines:

This file contains the main plotting fuctions for s3_ppxf_fitspectra.
"""

import os
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import massivepy.constants as const
import massivepy.spectrum as spec
import massivepy.io as mpio


def plot_s3_fullfit(plot_info):
    plot_path = plot_info['plot_path']

    # get data from fits files of ppxf fit output
    fitdata = mpio.get_friendly_ppxf_output(plot_info['main_output'])
    nbins = fitdata['nbins']
    nmoments = fitdata['nmoments']
    moment_names = ['h{}'.format(m+1) for m in range(nmoments)]
    moment_names[0] = 'V'
    moment_names[1] = 'sigma'
    # get spectrum and bin information
    specset = spec.read_datacube(plot_info['binspectra_path'])
    specset = specset.get_subset(fitdata['bins']['id'])
    if plot_info['run_type']=='bins':
        bininfo = np.genfromtxt(plot_info['bininfo_path'],names=True,
                                skip_header=1)
        ibins_all = {int(bininfo['binid'][i]):i for i in range(len(bininfo))}
        ibins = [ibins_all[binid] for binid in fitdata['bins']['id']]

    # save "friendly" text output for theorists
    txtfile_header = 'Columns are as follows:'
    colnames = fitdata['temps'].dtype.names
    txtfile_header += '\n ' + ' '.join(colnames)
    txtfile_header += '\n{} nonzero templates out of {}'.format(
        len(fitdata['temps']),fitdata['ntemps'])
    fmt = ['%i']
    fmt.extend(['%-8g']*(len(colnames)-1))
    np.savetxt(plot_info['temps_output'],fitdata['temps'],fmt=fmt,
               header=txtfile_header,delimiter='\t')

    # prep comparison plot info, if available
    if not plot_info['compare_moments']=='none':
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

    # template plots
    catalogfile = os.path.join(plot_info['templates_dir'],'catalog.txt')
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
    target_specset = specset.crop(plot_info['fit_range'])
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
    for m in plot_info['mask']:
        ax.axvspan(m[0],m[1],fc='k',ec='none',alpha=0.5,lw=0)
    ax.set_xlabel('wavelength ({})'.format("units"))
    ax.set_ylabel('bin number')
    ax.set_yticks([0])
    ax.autoscale(tight=True)
    pdf.savefig(fig)
    plt.close(fig)
    
    pdf.close()
    return

def plot_s3_binfit(plot_info):
    plot_path = plot_info['plot_path']

    # get data from fits files of ppxf fit output
    fitdata = mpio.get_friendly_ppxf_output(plot_info['main_output'])
    nbins = fitdata['nbins']
    nmoments = fitdata['nmoments']
    moment_names = ['h{}'.format(m+1) for m in range(nmoments)]
    moment_names[0] = 'V'
    moment_names[1] = 'sigma'
    # get spectrum and bin information
    specset = spec.read_datacube(plot_info['binspectra_path'])
    specset = specset.get_subset(fitdata['bins']['id'])
    if plot_info['run_type']=='bins':
        bininfo = np.genfromtxt(plot_info['bininfo_path'],names=True,
                                skip_header=1)
        ibins_all = {int(bininfo['binid'][i]):i for i in range(len(bininfo))}
        ibins = [ibins_all[binid] for binid in fitdata['bins']['id']]

    if os.path.isfile(plot_info['mc_output']):
        have_mc = True
        mcdata = mpio.get_friendly_ppxf_output_mc(plot_info['mc_output'])
    else:
        have_mc = False

    # save "friendly" text output for theorists
    txtfile_array = np.zeros((nbins,1+2*nmoments))
    txtfile_header = 'Fit results for {}'.format(plot_info['gal_name'])
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
    np.savetxt(plot_info['moments_output'],txtfile_array,fmt=fmt,
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
    if not plot_info['compare_moments']=='none':
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
    
    # plot each spectrum, y-axis also represents bin number
    figheight = max(fitdata['bins']['id'])
    figheight = max(figheight,4)
    fig = plt.figure(figsize=(6, figheight))
    fig.suptitle('bin spectra by bin number')
    ax = fig.add_axes([0.05,0.05,0.9,0.9])
    target_specset = specset.crop(plot_info['fit_range'])
    for i,binid in enumerate(fitdata['bins']['id']):
        spectrum = target_specset.get_subset([binid]).spectra[0]
        spectrum = spectrum/np.median(spectrum)
        waves = target_specset.waves
        model = fitdata['spec']['bestmodel'][i]
        modelwaves = fitdata['waves']
        # modelwaves should be same as waves, but is longer by one pixel!!
        # this bug shows up only (so far) in NGC1129
        ax.plot(waves,binid-spectrum+spectrum[0],c='k')
        ax.plot(modelwaves,binid-model+spectrum[0],c='r',lw=0.7)
        ax.text(waves[0],binid-0.4,
                r'$\chi^2={:4.2f}$'.format(fitdata['bins']['chisq'][i]))
    # find regions to mask
    # should add masking of bad_data as well!
    for m in plot_info['mask']:
        ax.axvspan(m[0],m[1],fc='k',ec='none',alpha=0.5,lw=0)
    ax.set_xlabel('wavelength ({})'.format("units"))
    ax.set_ylabel('bin number')
    ax.autoscale(tight=True)
    ax.set_ylim(ymin=-2,ymax=max(fitdata['bins']['id'])+1)
    ax.invert_yaxis()
    ax.tick_params(labeltop='on',top='on')
    pdf.savefig(fig)
    plt.close(fig)
    
    pdf.close()
    return
