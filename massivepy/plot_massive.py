"""
MASSIVE-specific plotting routines
"""

import functools

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches

import massivepy.constants as const
import massivepy.spectralresolution as res
import massivepy.IFUspectrum as ifu
import massivepy.gausshermite as gh


def colormap_setup(x,cmap,logsafe='off'):
    """
    Generate everything needed to plot a scalar field x using a colormap.
    Args:
      x - the quantity you want to plot
      cmap - the colormap you want to use (designed for built-ins like 'Reds',
             but can probably be used for other ways of defining colormaps)
      logsafe - what to do about zero and negative values in the log scale.
                'off' does nothing, will break if negative values are present
                'max' sets all bad values to the max value
    Returns:
      colors - a dict of containing all the relevant items:
                x - the original data, after "logsafe" applied
                vmin - minimum value, after "logsafe" applied
                vmax - maximum value, after "logsafe" applied
                x_norm, vmin_norm, vmax_norm - same, with norm applied
                mappable - ScalarMappable object for use in plotting
                c - rgba colors for plotting, same length as x
    """
    colors = {}
    if logsafe=='off':
        pass
    elif logsafe=='max':
        x = np.where(x>0,x,max(x)*np.ones(x.shape))
    else:
        raise Exception('Invalid choice of logsafe in colormap_setup.')
    colors['x'] = x.copy()
    colors['vmin'] = min(x)
    colors['vmax'] = max(x)
    norm = mpl.colors.LogNorm(vmin=colors['vmin'],vmax=colors['vmax'])
    colors['x_norm'] = norm(x)
    colors['vmin_norm'] = norm(colors['vmin'])
    colors['vmax_norm'] = norm(colors['vmax'])
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([colors['vmin'],colors['vmax']])
    colors['mappable'] = mappable
    c = mappable.to_rgba(x)
    colors['c'] = c
    return colors

def lin_colormap_setup(x,cmap):
    colors = {}
    colors['x'] = x.copy()
    colors['vmin'] = min(x)
    colors['vmax'] = max(x)
    #norm = mpl.colors.LogNorm(vmin=colors['vmin'],vmax=colors['vmax'])
    #colors['x_norm'] = norm(x)
    #colors['vmin_norm'] = norm(colors['vmin'])
    #colors['vmax_norm'] = norm(colors['vmax'])
    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array([colors['vmin'],colors['vmax']])
    colors['mappable'] = mappable
    c = mappable.to_rgba(x)
    colors['c'] = c
    return colors

def scalarmap(figtitle='default figure title',
              xlabel='default xaxis label', ylabel='default yaxis label',
              figsize=(6,6), ax_loc=[0.15,0.1,0.7,0.7],
              axC_loc=[0.15,0.8,0.7,0.8], axC_mappable=None,
              axC_label='default colorbar label'):
    """
    Generates figure and axes for a typical scalar map (fibermaps, binmaps).
    Default axes is already square, so that coordinates come out as desired,
    but it will allow you to break this.
    If axC_mappable is provided, also does a colormap. Note this puts nice
    minor ticks in assuming the colormap uses a log scale - this should be
    made more smart/flexible.
    """
    fig = plt.figure(figsize=figsize)
    fig.suptitle(figtitle)
    ax = fig.add_axes(ax_loc)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if not axC_mappable is None:
        axC = fig.add_axes(axC_loc)
        axC.set_visible(False)
        # futz with minor ticks only for LogNorm case
        if isinstance(axC_mappable.norm,mpl.colors.LogNorm):
            kw = {'ticks':mpl.ticker.LogLocator(subs=range(10))}
        else:
            kw = {}
        cb = fig.colorbar(axC_mappable,ax=axC,label=axC_label,
                          orientation='horizontal',**kw)
        ticks = axC_mappable.norm.inverse(cb.ax.xaxis.get_majorticklocs())
        cb.set_ticks(ticks)
        ticklabels = ['' for t in ticks]
        ticklabels[0] = ticks[0]
        ticklabels[-1] = ticks[-1]
        cb.set_ticklabels(ticklabels)
    return fig, ax
