"""
Process the raw Mitchell datacubes into a format more accessible
for binning and fitting.

The changes are:
 - Coordinates converted from (RA, DEC) to projected Cartesian arcsec
 - Arc frames are fit by Gaussian line profiles and replaced with the
   resulting samples of fwhm(lambda) for each fiber

input:
  takes one command line argument, a path to the input parameter text file
  s1_process_mitchell_params_example.txt is an example
  can take multiple parameter files if you want to process multiple galaxies
  (give one param file per galaxy)

output:
  one processed datacube for each input raw datacube
  one pdf with diagnostic plots
"""


import argparse
import re
import os

import numpy as np
import pandas as pd
import shapely.geometry as geo
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import astropy.io.fits as fits

import utilities as utl
import massivepy.constants as const
import massivepy.spectralresolution as res
import massivepy.IFUspectrum as ifu
import massivepy.io as mpio
import massivepy.gausshermite as gh

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
    # parse input parameter file
    output_dir, gal_name = mpio.parse_paramfile_path(paramfile_path)
    input_params = utl.read_dict_file(paramfile_path)
    raw_cube_path = input_params['raw_mitchell_cube']
    target_positions = pd.read_csv(input_params["target_positions"],
                                   comment='#', sep="[ \t]+",
                                   engine='python')
    run_name = input_params['run_name']
    # construct output file names
    output_filename = "{}-s1-{}-mitchellcube.fits".format(gal_name,run_name)
    plot_filename = "{}-s1-{}-mitchellcube.pdf".format(gal_name,run_name)
    ir_filename = "{}-s1-{}-ir.txt".format(gal_name,run_name)
    ir_fitsfilename = "{}-s1-{}-ir.fits".format(gal_name,run_name)
    output_path = os.path.join(output_dir, output_filename)
    ir_path = os.path.join(output_dir, ir_filename)
    ir_fitspath = os.path.join(output_dir, ir_fitsfilename)
    plot_path = os.path.join(output_dir, plot_filename)
    # save relevant info for plotting to a dict
    plot_info = {'data_path': output_path, 'plot_path': plot_path,
                 'ir_path': ir_path, 'raw_cube_path': raw_cube_path,
                 'targets_path': input_params['target_positions'],
                 'gal_name':gal_name}
    things_to_plot.append(plot_info)

    # decide whether to continue with script or skip to plotting
    if os.path.isfile(output_path):
        if input_params['skip_rerun']=='yes':
            print '\nSkipping re-run of {}, plotting only'.format(gal_name)
            continue
        elif input_params['skip_rerun']=='no':
            print '\nRunning {} again, will overwrite output'.format(gal_name)
        else:
            raise Exception("skip_rerun must be yes or no")

    # start processing
    data, headers = utl.fits_quickread(raw_cube_path)
    spec_unit = const.flux_per_angstrom  # assume spectrum units
    wave_unit = const.angstrom  # assume wavelength units
    # get data
    gal_position = target_positions[target_positions.Name == gal_name]
    gal_center = gal_position.Ra.iat[0], gal_position.Dec.iat[0]
    gal_pa = gal_position.PA_best.iat[0]
        # .ita[0] extracts scalar value from a 1-element dataframe
    print "\n{}".format(gal_name)
    print "  raw datacube: {}".format(raw_cube_path)
    print "        center: {}, {}".format(*gal_center)
    print "            pa: {}".format(gal_pa)
    try:
        # wavelengths of arc spectra are specifically included
        spectra, noise, all_waves, coords, arcs, all_inst_waves = data
        spectra_h, noise_h, waves_h, coords_h, arcs_h, inst_waves_h = headers
        gal_waves = all_waves[0, :]  # assume uniform samples; gal rest frame
        inst_waves = all_inst_waves[0, :]  # instrument rest frame
        redshift = waves_h['z']  # assumed redshift of galaxy
    except ValueError:
        # wavelength of arc spectra not included - compute by shifting
        # the spectra wavelength back into the instrument rest frame
        spectra, noise, all_waves, coords, arcs = data
        spectra_h, noise_h, waves_h, coords_h, arcs_h = headers
        gal_waves = all_waves[0, :]  # assume uniform samples; gal rest frame
        redshift = waves_h['z']  # assumed redshift of galaxy
        inst_waves = gal_waves*(1 + redshift)  # instrument rest frame
    print "  fitting arc frames..."
    spec_res_samples = res.fit_arcset(inst_waves, arcs,
                                      const.mitchell_arc_centers,
                                      const.mitchell_nominal_spec_resolution)
    # NEW PLAN: save the IR in a simple text file, don't copy the cube
    nfibers, nlines, ncols = spec_res_samples.shape
    ir_header = "Fits of arc frames for each fiber"
    ir_header += "\n interpolated from {} arc lamp lines".format(nlines)
    ir_header += "\n reported in instrument rest frame"
    ir_header += "\nFour columns for each line, as follows:"
    ir_header += "\n fiducial center, fit center, fit fwhm, fit height"
    ir_savearray = np.zeros((nfibers, 4*nlines))
    fmt = nlines*['%-7.6g','%-7.6g','%-7.4g','%-7.4g']
    ir_savearray[:,0::4] = const.mitchell_arc_centers
    ir_savearray[:,1::4] = spec_res_samples[:,:,0]
    ir_savearray[:,2::4] = spec_res_samples[:,:,1]
    ir_savearray[:,3::4] = spec_res_samples[:,:,2]
    np.savetxt(ir_path, ir_savearray, fmt=fmt, delimiter='\t', header=ir_header)


for plot_info in things_to_plot:
    print plot_info['data_path']

    #Collect data to plot
    ifuset = ifu.read_raw_datacube(plot_info['raw_cube_path'],
                                   plot_info['targets_path'],
                                   plot_info['gal_name'],
                                   ir_path=plot_info['ir_path'])
    print ifuset.spectrumset.comments
    nfibers = ifuset.spectrumset.comments['nfibers']
    fibersize = ifuset.linear_scale
    xcoords = -ifuset.coords[:,0] #Flipping east-west to match pictures
    ycoords = ifuset.coords[:,1]
    squaremax = np.amax(np.abs(ifuset.coords)) + fibersize
    rcoords = np.sqrt(xcoords**2 + ycoords**2)
    coordunit = ifuset.coords_unit
    #The first quantity we want is flux per fiber
    logfluxes = np.log10(ifuset.spectrumset.compute_flux())
    logfmax = max(logfluxes)
    logfmin = min(logfluxes)
    fluxunit = ifuset.spectrumset.integratedflux_unit
    fcmap = plt.cm.get_cmap('Reds')
    fgetcolor = lambda f: fcmap((f - logfmin)/(logfmax-logfmin))
    #The second quantity we want is s2n per fiber
    logs2n = np.log10(ifuset.spectrumset.compute_mean_s2n())
    logsmax = max(logs2n)
    logsmin = min(logs2n)
    scmap = plt.cm.get_cmap('Greens')
    sgetcolor = lambda s: scmap((s - logsmin)/(logsmax-logsmin))

    #Will create 4 figures, to do flux and s2n in both map and vs-radius form
    fig1 = plt.figure(figsize=(6,6))
    fig1.suptitle('flux map')
    ax1 = fig1.add_axes([0.15,0.1,0.7,0.7])
    fig2 = plt.figure(figsize=(6,6))
    fig2.suptitle('s2n map')
    ax2 = fig2.add_axes([0.15,0.1,0.7,0.7])    
    fig3 = plt.figure(figsize=(6,6))
    fig3.suptitle('flux vs radius')
    ax3 = fig3.add_axes([0.15,0.1,0.7,0.7])
    fig4 = plt.figure(figsize=(6,6))
    fig4.suptitle('s2n vs radius')
    ax4 = fig4.add_axes([0.15,0.1,0.7,0.7])

    #Now loop over the fibers and plot things!
    for ifiber in range(nfibers):
        ax1.add_patch(patches.Circle((xcoords[ifiber],ycoords[ifiber]),
                                     fibersize,lw=0.25,
                                     fc=fgetcolor(logfluxes[ifiber])))
        ax1.text(xcoords[ifiber],ycoords[ifiber],
                 str(ifuset.spectrumset.ids[ifiber]),fontsize=5,
                 horizontalalignment='center',verticalalignment='center')
        ax2.add_patch(patches.Circle((xcoords[ifiber],ycoords[ifiber]),
                                     fibersize,lw=0.25,
                                     fc=sgetcolor(logs2n[ifiber])))
        ax2.text(xcoords[ifiber],ycoords[ifiber],
                 str(ifuset.spectrumset.ids[ifiber]),fontsize=5,
                 horizontalalignment='center',verticalalignment='center')
        ax3.text(rcoords[ifiber],logfluxes[ifiber],
                 str(ifuset.spectrumset.ids[ifiber]),fontsize=5,
                 horizontalalignment='center',verticalalignment='center')
        ax4.text(rcoords[ifiber],logs2n[ifiber],
                 str(ifuset.spectrumset.ids[ifiber]),fontsize=5,
                 horizontalalignment='center',verticalalignment='center')
    #Fix axes bounds
    ax1.axis([-squaremax,squaremax,-squaremax,squaremax])
    ax2.axis([-squaremax,squaremax,-squaremax,squaremax])
    ax3.axis([min(rcoords),max(rcoords),min(logfluxes),max(logfluxes)])
    ax4.axis([min(rcoords),max(rcoords),min(logs2n),max(logs2n)])
    #Make labels
    label_x = r'$\leftarrow$east ({}) west$\rightarrow$'.format(coordunit)
    label_y = r'$\leftarrow$south ({}) north$\rightarrow$'.format(coordunit)
    label_r = r'radius ({})'.format(coordunit)
    label_flux = r'flux (log 10 [{}])'.format(fluxunit)
    label_s2n = r's2n (log 10)'
    ax1.set_xlabel(label_x)
    ax1.set_ylabel(label_y)
    ax2.set_xlabel(label_x)
    ax2.set_ylabel(label_y)
    ax3.set_xlabel(label_r)
    ax3.set_ylabel(label_flux)
    ax4.set_xlabel(label_r)
    ax4.set_ylabel(label_s2n)
    #Do colorbars
    ax1C = fig1.add_axes([0.15,0.8,0.7,0.8])
    ax1C.set_visible(False)
    mappable_flux = plt.cm.ScalarMappable(cmap=fcmap)
    mappable_flux.set_array([logfmin,logfmax])
    fig1.colorbar(mappable_flux,orientation='horizontal',ax=ax1C,
                  label=label_flux)
    ax2C = fig2.add_axes([0.15,0.8,0.7,0.8])
    ax2C.set_visible(False)
    mappable_s2n = plt.cm.ScalarMappable(cmap=scmap)
    mappable_s2n.set_array([logsmin,logsmax])
    fig2.colorbar(mappable_s2n,orientation='horizontal',ax=ax2C,
                  label=label_s2n)

    #Assemble all into multipage pdf
    pdf = PdfPages(plot_info['plot_path'])
    pdf.savefig(fig1)
    pdf.savefig(fig2)
    pdf.savefig(fig3)
    pdf.savefig(fig4)

    # new ir testing plots!
    skipnumber = 40 #only do 1 of every skipnumber fibers
    # load up the arc frames - this should go as part of the ifu
    data, headers = utl.fits_quickread(plot_info['raw_cube_path'])
    try:
        # wavelengths of arc spectra are specifically included
        spectra, noise, all_waves, coords, arcs, all_inst_waves = data
        spectra_h, noise_h, waves_h, coords_h, arcs_h, inst_waves_h = headers
        gal_waves = all_waves[0, :]  # assume uniform samples; gal rest frame
        inst_waves = all_inst_waves[0, :]  # instrument rest frame
        redshift = waves_h['z']  # assumed redshift of galaxy
    except ValueError:
        # wavelength of arc spectra not included - compute by shifting
        # the spectra wavelength back into the instrument rest frame
        spectra, noise, all_waves, coords, arcs = data
        spectra_h, noise_h, waves_h, coords_h, arcs_h = headers
        gal_waves = all_waves[0, :]  # assume uniform samples; gal rest frame
        redshift = waves_h['z']  # assumed redshift of galaxy
        inst_waves = gal_waves*(1 + redshift)  # instrument rest frame
    waves = inst_waves
    spectra = arcs[::skipnumber]
    spectra = (spectra.T/np.max(spectra, axis=1)).T
    # load up the ir textfile
    ir_textinfo = np.genfromtxt(plot_info['ir_path'])
    ir_fidcenter = ir_textinfo[::skipnumber,0::4]
    ir_center = ir_textinfo[::skipnumber,1::4]
    ir_fwhm = ir_textinfo[::skipnumber,2::4]
    ir_height = ir_textinfo[::skipnumber,3::4]
    nplotfibers, nlines = ir_fidcenter.shape
    # do a figure for each line
    for i in range(nlines):
        fig = plt.figure(figsize=(6,6))
        fig.suptitle('ir testing')
        ax = fig.add_axes([0.15,0.1,0.7,0.7])
        for j in range(nplotfibers):
            startwave, endwave = ir_fidcenter[j,i]-20, ir_fidcenter[j,i]+20
            startpix, endpix = np.searchsorted(waves,[startwave,endwave])
            ax.plot(waves[startpix:endpix],spectra[j][startpix:endpix],
                    c='k',alpha=0.2)
            sigma = ir_fwhm[j,i]/const.gaussian_fwhm_over_sigma
            height = ir_height[j,i]/(2*np.pi)
            center = ir_center[j,i]
            model_p = [center,sigma]
            model = gh.unnormalized_gausshermite_pdf(waves[startpix:endpix],
                                                      model_p)*height
            # apparently height means nothing...
            model = model*max(spectra[j][startpix:endpix])/max(model)
            ax.plot(waves[startpix:endpix],model,c='b',alpha=0.2)
            if i==5:
                sigma2 = (const.mitchell_nominal_spec_resolution
                          /const.gaussian_fwhm_over_sigma)
                model2_p = [center,sigma2]
                model2 =gh.unnormalized_gausshermite_pdf(waves[startpix:endpix],
                                                         model2_p)*height
                model2 = model2*max(spectra[j][startpix:endpix])/max(model2)
                ax.plot(waves[startpix:endpix],model2,c='r',alpha=0.2)
        ax.set_ylim(ymin=0,ymax=0.004)
        pdf.savefig(fig)
        plt.close(fig)

    # now do the ir vs wavelength
    fig = plt.figure(figsize=(6,6))
    fig.suptitle('ir testing')
    ax = fig.add_axes([0.15,0.1,0.7,0.7])
    for i in range(nplotfibers):
        ax.plot(ir_center[i,:],ir_fwhm[i,:],c='r',alpha=0.2)
    pdf.savefig(fig)
    plt.close(fig)
    pdf.close()
