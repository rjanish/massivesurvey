"""
MASSIVE-specific plotting routines:

This file contains the main plotting fuctions for s4_ppxf_fitspectra.
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
import massivepy.postprocess as post
from plotting.geo_utils import polar_box


def plot_s4_postprocess(gal_name=None,binfit_path=None,plot_path=None,
                        rdata_path=None):
    # get data
    rdata = np.genfromtxt(rdata_path, names=True, skip_header=1)
    rmeta = mpio.read_friendly_header(rdata_path)
    binmoments = np.genfromtxt(binfit_path,names=True,skip_header=1)
    if not rmeta['junk bins']==0:
        binmoments = binmoments[:-rmeta['junk bins']]
    labels = {'lam': r'$\lambda_R$',
              'sigma':r'$\sigma$',
              'flux':'flux'}

    ### Plotting Begins! ###

    pdf = PdfPages(plot_path)

    # plot the actual lambda first
    fig = plt.figure(figsize=(6,6))
    fig.suptitle('{} {}'.format(gal_name,labels['lam']))
    ax = fig.add_axes([0.15,0.1,0.8,0.7])
    lam_keys = ['','_vmin','_vmax']
    lam_colors = ['k','b','g']
    lam_labels = ['fiducial','min V0','max V0']
    for k,c,l in zip(lam_keys,lam_colors,lam_labels):
        ax.plot(rdata['r'],rdata['lam_loc'+k],c=c,ls='-.')
        ax.plot(rdata['r_en'],rdata['lam_en'+k],c=c,ls='-',label=l)
    ax.plot(0,0,c='k',ls='-.',label='local')
    ax.axhline(rmeta['slow/fast cutoff'],c='k',ls='--',label='slow/fast')
    ax.axvline(rmeta['gal re'],c='k',ls=':',label=r'$R_e$')
    ax.plot(rmeta['gal re'],rmeta['lambda re'],ls='',marker='o',mfc='k')
    ax.plot(.5*rmeta['gal re'],rmeta['lambda half re'],ls='',marker='o',mfc='k')
    ax.set_xlabel('radius')
    ax.set_ylabel(labels['lam'])
    ax.legend(loc='lower center',bbox_to_anchor=(0.5,1),ncol=3)
    pdf.savefig(fig)
    plt.close(fig)


    fig, ax = mplt.scalarmap(figtitle='{} {}'.format(gal_name,labels['sigma']),
                             xlabel='radius',ylabel=labels['sigma'])
    ax.plot(rdata['r'],rdata['sig_loc'],c='b',ls='-.')
    ax.fill_between(rdata['r'],rdata['sig_loc']-rdata['sig_loc_err'],
                    rdata['sig_loc']+rdata['sig_loc_err'],alpha=0.2)
    ax.plot(rdata['r_en'],rdata['sig_en'],c='b')
    pdf.savefig(fig)
    plt.close(fig)


    fig, ax = mplt.scalarmap(figtitle='{} h3 correlation'.format(gal_name),
                             xlabel='V (scaled)',ylabel='h3')
    voversigma = (binmoments['V']-rmeta['v0_fiducial'])/binmoments['sigma']
    ax.plot(voversigma,binmoments['h3'],ls='',
            marker='o',mfc='c',ms=7,zorder=-1)
    for i in range(len(voversigma)):
        ax.text(voversigma[i],binmoments['h3'][i],str(i+1),fontsize=5,
                horizontalalignment='center',verticalalignment='center')
    fakeV = np.array([np.min(voversigma),np.max(voversigma)])
    ax.plot(fakeV,rmeta['h3 intercept']+fakeV*rmeta['h3 slope'])
    ax.fill_between(fakeV,
        rmeta['h3 intercept']+fakeV*(rmeta['h3 slope']-rmeta['h3 slope err']),
        rmeta['h3 intercept']+fakeV*(rmeta['h3 slope']+rmeta['h3 slope err']),
                    alpha=0.1)
    ax.axhline(0,c='k',lw=2)
    ax.axhline(rmeta['h3 average'],c='r',label='h3 average')
    ax.axhline(rmeta['h5 average'],c='m',label='h5 average')
    ax.legend(loc='lower center',bbox_to_anchor=(0.5,1),ncol=3)
    pdf.savefig(fig)
    plt.close(fig)


    fig, ax = mplt.scalarmap(figtitle='{} h4 correlation'.format(gal_name),
                             xlabel='sigma (scaled)',ylabel='h4')
    sigma0 = np.average(binmoments['sigma'])
    sigmaoversigma = (binmoments['sigma']-sigma0)/sigma0
    ax.plot(sigmaoversigma,binmoments['h4'],ls='',
            marker='o',mfc='c',ms=7,zorder=-1)
    for i in range(len(sigmaoversigma)):
        ax.text(sigmaoversigma[i],binmoments['h4'][i],str(i+1),fontsize=5,
                horizontalalignment='center',verticalalignment='center')
    fakeS = np.array([np.min(sigmaoversigma),np.max(sigmaoversigma)])
    ax.plot(fakeS,rmeta['h4 intercept']+fakeS*rmeta['h4 slope'])
    ax.fill_between(fakeS,
        rmeta['h4 intercept']+fakeS*(rmeta['h4 slope']-rmeta['h4 slope err']),
        rmeta['h4 intercept']+fakeS*(rmeta['h4 slope']+rmeta['h4 slope err']),
                    alpha=0.1)
    ax.axhline(0,c='k',lw=2)
    ax.axhline(rmeta['h4 average'],c='r',label='h4 average')
    ax.axhline(rmeta['h6 average'],c='m',label='h6 average')
    ax.legend(loc='lower center',bbox_to_anchor=(0.5,1),ncol=3)
    pdf.savefig(fig)
    plt.close(fig)

    pdf.close()
