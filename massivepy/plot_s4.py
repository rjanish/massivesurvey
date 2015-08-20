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
from plotting.geo_utils import polar_box


def plot_s4_postprocess(gal_name=None,plot_path=None,rprofiles_path=None):
    # get data
    rprofiles = np.genfromtxt(rprofiles_path, names=True, skip_header=1)
    labels = {'lam': r'$\lambda_R$',
              'sigma':r'$\sigma$',
              'flux':'flux'}
    ii = rprofiles['toplot'].astype(bool)

    ### Plotting Begins! ###

    pdf = PdfPages(plot_path)

    # plot the actual lambda first
    fig = plt.figure(figsize=(6,6))
    fig.suptitle(labels['lam'])
    ax = fig.add_axes([0.15,0.1,0.8,0.7])
    ax.plot(rprofiles['rencl'][ii],rprofiles['lam'][ii],
            c='b',label='fiducial')
    ax.plot(rprofiles['rencl'][ii],rprofiles['lam_med'][ii],
            c='c',label='median for V0')
    ax.plot(rprofiles['rencl'][ii],rprofiles['lam_fluxw'][ii],
            c='r',label='flux weighted')
    ax.set_xlabel('radius')
    ax.set_ylabel(labels['lam'])
    ax.legend(loc='lower center',bbox_to_anchor=(0.5,1),ncol=2)
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
    pdf.savefig(fig)
    plt.close(fig)

    pdf.close()
    return
