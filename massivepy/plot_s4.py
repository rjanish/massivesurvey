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


def plot_s4_postprocess(gal_name=None,plot_path=None,rprofiles_path=None):
    # get data
    rprofiles = np.genfromtxt(rprofiles_path, names=True, skip_header=1)
    rmeta = mpio.read_friendly_header(rprofiles_path)
    labels = {'lam': r'$\lambda_R$',
              'sigma':r'$\sigma$',
              'flux':'flux'}
    ii = rprofiles['toplot'].astype(bool)
    jj = rprofiles['diff_toplot'].astype(bool)

    ### Plotting Begins! ###

    pdf = PdfPages(plot_path)

    # plot the actual lambda first
    fig = plt.figure(figsize=(6,6))
    fig.suptitle(labels['lam'])
    ax = fig.add_axes([0.15,0.1,0.8,0.7])
    lam_keys = ['lam','lam_minv0','lam_maxv0']
    lam_colors = ['k','b','g']
    lam_labels = ['fiducial','min V0','max V0']
    for k,c,l in zip(lam_keys,lam_colors,lam_labels):
        ax.plot(rprofiles['rencl'][ii],rprofiles[k][ii],c=c,label=l)
        ax.plot(rprofiles['rencl'][jj],rprofiles['diff_'+k][jj],c=c,ls='-.')
    ax.plot(0,0,c='k',ls='-.',label='differential')
    ax.axhline(rmeta['slow/fast cutoff'],c='k',ls='--',label='slow/fast')
    ax.axvline(rmeta['gal re'],c='k',ls=':',label=r'$R_e$')
    ax.plot(rmeta['gal re'],rmeta['lambda re'],ls='',marker='o',mfc='k')
    ax.plot(.5*rmeta['gal re'],rmeta['lambda half re'],ls='',marker='o',mfc='k')
    ax.set_xlabel('radius')
    ax.set_ylabel(labels['lam'])
    ax.legend(loc='lower center',bbox_to_anchor=(0.5,1),ncol=3)
    pdf.savefig(fig)
    plt.close(fig)


    fig, ax = mplt.scalarmap(figtitle=r'$V_0$ comparisons',
                             xlabel=r'choice of $V_0$',ylabel=r'$V_0$')
    v0_keys = ['v0_full0','v0_full-1','v0_full-2',
               'v0_binavg','v0_wbinavg','v0_binmed','v0_wbinmed','v0_fiducial']
    for k in v0_keys:
        if k not in rmeta: rmeta[k] = np.nan
    v0_labels = ['all fibers fullbin','binned fibers fullbin',
                 'symmetrical fullbin','bins average','weighted bins avg',
                 'bins median','weighted bins med','fiducial']
    ax.plot([rmeta[k] for k in v0_keys],marker='o')
    ax.set_xlim(xmin=-0.2,xmax=len(v0_keys)-0.8)
    ax.set_xticks(range(len(v0_keys)))
    ax.tick_params(labeltop='on',top='on',labelbottom='off')
    ax.set_xticklabels(v0_labels,rotation=20, ha='left')
    pdf.savefig(fig)
    plt.close(fig)

    fig = plt.figure(figsize=(6,6))
    fig.suptitle(labels['lam'])
    ax = fig.add_axes([0.15,0.1,0.8,0.7])
    ax.plot(rprofiles['rencl'],rprofiles['lam'],
            c='b',marker='s',ms=3,label='all points')
    ax.plot(rprofiles['rencl'][ii],rprofiles['lam'][ii],
            c='c',marker='o',ms=3,label='one per annulus')
    ax.set_xlabel('radius')
    ax.set_ylabel(labels['lam'])
    ax.legend(loc='lower center',bbox_to_anchor=(0.5,1),ncol=2)
    pdf.savefig(fig)
    plt.close(fig)


    fig, ax = mplt.scalarmap(figtitle=labels['sigma'],
                             xlabel='radius',ylabel=labels['sigma'])
    ax.plot(rprofiles['rencl'][ii],rprofiles['sig'][ii],c='b')
    ax.plot(rprofiles['rencl'][jj],rprofiles['diff_sig'][jj],c='b',ls='-.')
    pdf.savefig(fig)
    plt.close(fig)

    pdf.close()
    return
