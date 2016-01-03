"""
Plot the three brightest fibers of every set of s2 output.
Gonna inspect how the shifting is going.
"""

import os
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches

import massivepy.io as mpio
import massivepy.spectrum as spec
import utilities as utl


# get cmd line arguments
parser = argparse.ArgumentParser(description=__doc__,
                formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("paramfiles", nargs='*', type=str,
                    help="path(s) to input parameter file(s)")
args = parser.parse_args()
all_paramfile_paths = args.paramfiles

# empty list for outputs to plot
things_to_plot = []

for paramfile_path in all_paramfile_paths:
    print '\n\n====================================='
    # parse input parameter file, generate other needed paths
    output_dir, gal_name = mpio.parse_paramfile_path(paramfile_path)
    input_params = utl.read_dict_file(paramfile_path)

    fitspath_maker = lambda run_name: os.path.join(output_dir,
                        "{}-s2-{}-binspectra.fits".format(gal_name,run_name))

    run_names = input_params['run_name']
    nruns = len(run_names)
    fitspaths = [fitspath_maker(r) for r in run_names]

    ndithers = 3 ### should really do something better with this?

    check = mpio.pathcheck(fitspaths,nruns*['.fits'],gal_name)
    if not check:
        print 'Something is wrong with the input paths for {}'.format(gal_name)
        print 'Skipping to next galaxy.'
        continue

    ########
    # Do some plotting
    ########

    plot_path = os.path.join(output_dir,"{}-s2X-allruns.pdf".format(gal_name))

    ### Plotting Begins! ###

    pdf = PdfPages(plot_path)

    specsets = [spec.read_datacube(path) for path in fitspaths]

    # look at the spectra themselves, and the noise
    figs, axs, yticks, yticklabels = {}, {}, {}, {}
    for key in ['spectra','noise']:
        figs[key] = plt.figure(figsize=(6,6))
        figs[key].suptitle(key)
        axs[key] = figs[key].add_axes([0.1,0.1,0.85,0.75])
        yticks[key], yticklabels[key] = [], []
    axs['spectra'].set_xlabel('wavelength')
    axs['spectra'].set_ylabel('flux (bin number)')
    axs['noise'].set_xlabel('wavelength')
    axs['noise'].set_ylabel('flux (bin number)')

    runcolors = ['b','g','r']
    plotkw = {'marker':'.','ms':2,'lw':0.5,'alpha':0.6}
    handles, labels = [], []
    for i in range(nruns):
        waves = specsets[i].waves
        plotkw['color'] = runcolors[i]
        handles.append(patches.Patch(color=runcolors[i]))
        labels.append(run_names[i])
        for j in range(ndithers):
            spectrum = specsets[i].spectra[j,:]
            spectrum = spectrum/np.median(spectrum)
            noise = specsets[i].metaspectra['noise'][j,:]
            ii = (waves > 3950) & (waves < 4000)
            axs['spectra'].plot(waves[ii],0.1*j+spectrum[ii],**plotkw)
            axs['noise'].plot(waves[ii],0.1*j+noise[ii],**plotkw)
            if i==0:
                yticks['spectra'].append(0.1*j+spectrum[ii][0])
                yticklabels['spectra'].append(specsets[i].ids[j])
                yticks['noise'].append(0.1*j+noise[ii][0])
                yticklabels['noise'].append(specsets[i].ids[j])

    for key in ['spectra','noise']:
        axs[key].legend(handles,labels,bbox_to_anchor=[0.5,1.01],
                        loc='lower center',ncol=3,fontsize=10)
        axs[key].set_yticks(yticks[key])
        axs[key].set_yticklabels(yticklabels[key])
        pdf.savefig(figs[key])
        plt.close(figs[key])            

    
    pdf.close()
    
