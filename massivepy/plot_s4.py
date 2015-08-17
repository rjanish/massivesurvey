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


def plot_s4_postprocess(gal_name=None,plot_path=None,lambda_path=None,
                        lambda_path_med=None,lambda_path_jstyle=None,
                        fits_path=None,rprofiles_path=None):
    # get data
    rprofiles = np.genfromtxt(rprofiles_path, names=True)
    #lamR = np.genfromtxt(lambda_path,names=True)
    #lamR_median = np.genfromtxt(lambda_path_med,names=True)
    #lamR_jstyle = np.genfromtxt(lambda_path_jstyle,names=True)
    labels = {'R': 'radius',
              'Vavg': r'$\langle |V| \rangle$',
              'RVavg': r'$\langle R |V| \rangle$',
              'm2avg': r'$\langle \sqrt{V^2 + \sigma^2} \rangle$',
              'Rm2avg': r'$\langle R \sqrt{V^2 + \sigma^2} \rangle$',
              'lam': r'$\lambda_R$',
              'V':r'$V$',
              'Vraw':r'$V_{\rm raw}$',
              'sigma':r'$\sigma$',
              'flux':'flux'}

    ### Plotting Begins! ###

    pdf = PdfPages(plot_path)

    # plot the actual lambda first
    fig = plt.figure(figsize=(6,5))
    fig.suptitle(labels['lam'])
    ax = fig.add_axes([0.17,0.15,0.7,0.7])
    ax.plot(rprofiles['rbin'],rprofiles['lam'],c='b',label='flux avg')
    ax.plot(rprofiles['rbin'],rprofiles['lam_med'],c='c',label='median')
    ax.plot(rprofiles['rbin'],rprofiles['lam_old'],c='r',label='old median')
    #ax.plot(lamR_jstyle['R'],lamR_jstyle['lam'],c='r',label='jstyle')
    ax.set_xlabel('radius')
    ax.set_ylabel(labels['lam'])
    #if gal_name in ['NGC0057','NGC0507']:
    #    Jf = os.path.join(os.path.dirname(plot_path),'jenny.txt')
    #    Jthings = np.genfromtxt(Jf,names=True)
    #    ax.plot(Jthings['rad'],Jthings['lam'],c='m',label='Jenny')
    ax.legend()
    pdf.savefig(fig)
    plt.close(fig)

    fig = plt.figure(figsize=(6,5))
    fig.suptitle(labels['lam'])
    ax = fig.add_axes([0.17,0.15,0.7,0.7])
    ax.plot(rprofiles['rbin'],rprofiles['lam'],c='b',marker='s',label='all')
    ii = rprofiles['rtoplot'].astype(bool)
    ax.plot(rprofiles['rbin'][ii],rprofiles['lam'][ii],c='c',marker='o',label='good')
    ax.set_xlabel('radius')
    ax.set_ylabel(labels['lam'])
    ax.legend()
    pdf.savefig(fig)
    plt.close(fig)

    '''
    # plot the intermediate steps as subplots on one page
    fig = plt.figure(figsize=(6,6))
    fig.suptitle(r"Intermediate Steps (x-axis matching $\lambda_R$)")
    for i, thing in enumerate(['Vavg','RVavg','m2avg','Rm2avg']):
        ax = fig.add_subplot(2,2,i+1)
        ax.plot(lamR['R'],lamR[thing],c='b')
        ax.plot(lamR_median['R'],lamR_median[thing],c='c')
        ax.plot(lamR_jstyle['R'],lamR_jstyle[thing],c='r')
        if thing=='RVavg':
            ax.plot(Jthings['rad'],Jthings['num'],c='m')
            pass
        elif thing=='Rm2avg':
            ax.plot(Jthings['rad'],Jthings['denom'],c='m')
            pass
        ax.set_yscale('log')
        ax.set_title(labels[thing])
        ax.set_xticklabels([])
    pdf.savefig(fig)
    plt.close(fig)
    '''

    fig, ax = mplt.scalarmap(figtitle='sigma stuff',xlabel='',ylabel='')
    ax.plot(rprofiles['rbin'],rprofiles['sig'],c='b')
    #if gal_name in ['NGC0057','NGC0507']:
        # should already have Jthings
    #    ax.plot(Jthings['rad'],Jthings['sig_weighted'],c='m')
    pdf.savefig(fig)
    plt.close(fig)

    pdf.close()
    return
