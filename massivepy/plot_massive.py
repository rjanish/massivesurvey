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

def lin_colormap_setup(x,cmap,center=False):
    colors = {}
    colors['x'] = x.copy()
    if center:
        colors['vmin'] = -max(np.abs(x))
        colors['vmax'] = max(np.abs(x))
    else:
        colors['vmin'] = min(x)
        colors['vmax'] = max(x)
    mappable = plt.cm.ScalarMappable(cmap=cmap)
    mappable.set_array([colors['vmin'],colors['vmax']])
    mappable.set_clim(vmin=colors['vmin'],vmax=colors['vmax'])
    colors['mappable'] = mappable
    c = mappable.to_rgba(x)
    colors['c'] = c
    return colors

def scalarmap(figtitle='default figure title',
              xlabel='default xaxis label', ylabel='default yaxis label',
              figsize=(6,6), ax_loc=[0.15,0.1,0.7,0.7],
              axC_loc=[0.15,0.8,0.7,0.8], axC_mappable=None,
              axC_label='default colorbar label',
              axC_nticks=3):
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
        try:
            ticks = axC_mappable.norm.inverse(cb.ax.xaxis.get_majorticklocs())
            cb.set_ticks(ticks)
            ticklabels = ['' for t in ticks]
            skipticks = (len(ticklabels)-1)/axC_nticks + 1
            ticklabels[::skipticks] = ticks[::skipticks]
            cb.set_ticklabels(ticklabels)
        except:
            print "Your crappy tick stuff broke, finish your better code."
    return fig, ax

def tick_magic(xmin=0,xmax=1,nmin=3,nmax=6):
    """
    Home-brewed tick logic, because pyplot frustrates me.
    Optimal placement of ticks is determined by shortness of label; e.g. ticks
     will never be something like 1, 2.5, 4, 5.5, 7. Instead, they will be
     something like 1, 3, 5, 7.
    INPUT:
      -xmin, xmax: the beginning and end of the axis range
      -nmin, nmax: are the min and max allowed numbers of ticks
    OUTPUT (packaged as one dict containing the following):
      -ticks: locations of major ticks
      -ticklabels: tick labels, making smartest use of scientific notation
       (smarter ticklabels not implemented yet!)
      -minorticks: optional minor ticks, always at some power of 10.
    """
    xdiff = abs(xmax-xmin) # size of axis range
    xscale = 10**np.floor(np.log10(xdiff)) # basic tick interval
    nstart = np.ceil(xmin/xscale) # first tick is at nstart*xscale
    nstop = np.floor(xmax/xscale) # last tick is at nstop*xscale
    nticks = nstop-nstart+1 # number of ticks will need to be made sensible
    if nticks < nmin: # go down one power of 10 if needed
        xscale = xscale/10.0
        nstart = np.ceil(xmin/xscale)
        nstop = np.floor(xmax/xscale)
        nticks = nstop-nstart+1
    ticks = np.linspace(xscale*nstart,xscale*nstop,num=nticks)
    minorticks = ticks.copy() # minorticks is now set
    if nticks > nmax: # pick out subset of ticks if needed
        nnew, nskip = nticks, 1 # nnew is what nticks will become
        while nnew > nmax: # find smallest nskip that gives nnew <= nmax
            nskip += 1
            nnew = np.ceil(nticks/float(nskip))
        nextra = (nticks-1) % nskip # number of ticks taken off the end
        istart = nextra/2 # remove half the extra ticks from the front
        if (nextra%2==1) and ((xscale*nstart-xmin) < (xmax-xscale*nstop)):
            istart += 1 # be smart about where to take "odd" extra tick from
        ticks = ticks[istart::nskip] # length of new ticks equals nnew
    ticklabels = [str(t) for t in ticks] # make this smarter someday
    magic = {'ticks':ticks,'minorticks':minorticks,'ticklabels':ticklabels}
    return magic

def log_tick_magic(xmin,xmax,nmin=1,nmax=6):
    """
    Home-brewed tick logic, because pyplot frustrates me.
    Log axes become awkward in pyplot when the dynamic range is small, and
      the tick labeling is not optimal.
    INPUT:
      -xmin, xmax: the beginning and end of the axis range
      -nmin, nmax: are the min and max allowed numbers of ticks
    OUTPUT (packaged as one dict containing the following):
      -ticks: locations of major ticks
      -ticklabels: tick labels with smarter labels at 1, 10, 100
      -minorticks: either the usual log scale minor ticks or the "skipped"
       major ticks in cases where the dynamic range is large
    """
    nstart = int(np.ceil(np.log10(xmin))) # first tick is at 10^nstart
    nstop = int(np.floor(np.log10(xmax))) # last tick is at 10^nstop
    nticks = nstop-nstart+1 # number of ticks will need to be made sensible
    ticks = 10**np.linspace(nstart,nstop,num=nticks)
    minorticksections = []
    for i in range(nstart-1,nstop+1):
        section = np.array([2,3,4,5,6,7,8,9])*10**i
        section = section[(section>xmin)&(section<xmax)]
        minorticksections.append(section)
    minorticks = np.concatenate(minorticksections)
    if nticks < nmin: # if no power of 10 is in the range, use the minor ticks
        ticks = minorticks.copy()
        minorticks = []
        # now need to cleverly pick subset if some are too crowded
        # because the standard method below ignores log "squishing"
        # probably want to return in this section since labels will also change
    if nticks > nmax: # pick out subset of ticks if needed
        nnew, nskip = nticks, 1 # nnew is what nticks will become
        while nnew > nmax: # find smallest nskip that gives nnew <= nmax
            nskip += 1
            nnew = np.ceil(nticks/float(nskip))
        nextra = (nticks-1) % nskip # number of ticks taken off the end
        istart = nextra/2 # remove half the extra ticks from the front
        if (nextra%2==1) and ((xscale*nstart-xmin) < (xmax-xscale*nstop)):
            istart += 1 # be smart about where to take "odd" extra tick from
        minorticks = ticks.copy()
        ticks = ticks[istart::nskip] # length of new ticks equals nnew
    ticklabels = [r'$10^{:d}$'.format(n) for n in np.log10(ticks)]
    magic = {'ticks':ticks,'minorticks':minorticks,'ticklabels':ticklabels}
    return magic
