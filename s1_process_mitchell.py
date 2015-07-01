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

import utilities as utl
import massivepy.constants as const
import massivepy.spectralresolution as res
import massivepy.IFUspectrum as ifu
import massivepy.io as mpio

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
    output_path = os.path.join(output_dir, output_filename)
    plot_path = os.path.join(output_dir, plot_filename)
    # save relevant info for plotting to a dict
    plot_info = {'data_path': output_path, 'plot_path': plot_path}
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
    fiber_radius = const.mitchell_fiber_radius.value  # arcsec
    fiber_circle = lambda center: geo.Point(center).buffer(fiber_radius)
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
    print "  re-scaling coordinates..."
    cart_coords = ifu.center_coordinates(coords, gal_center)
    print "  fitting arc frames..."
    spec_res_samples = res.fit_arcset(inst_waves, arcs,
                                      const.mitchell_arc_centers,
                                      const.mitchell_nominal_spec_resolution)
    spec_res_full = np.nan*np.ones(spectra.shape)
    print "  interpolating spectral resolution..."
    for fiber_iter, fiber_res_samples in enumerate(spec_res_samples):
        galframe_samples = fiber_res_samples/(1 + redshift)
        gal_interp_func = utl.interp1d_constextrap(*galframe_samples.T)
        spec_res_full[fiber_iter] = gal_interp_func(gal_waves)
            # This scales the ir into the galaxy rest frame
    print ("cropping to {}-{} A (galaxy rest frame)..."
           "".format(*const.mitchell_crop_region))
    valid = utl.in_linear_interval(gal_waves, const.mitchell_crop_region)
    # TO DO: actual masking of sky lines, check for nonsense values
    bad_data = np.zeros(spectra.shape, dtype=bool)
    # save data
    fiber_numbers = np.arange(spectra.shape[0], dtype=int)
    vhelio = spectra_h["VHELIO"]
    comments = {
        "target":gal_name,
        "heliocentric correction applied":"{} [km/s]".format(vhelio),
        "wavelengths":"wavelength in galaxy rest frame",
        "applied galaxy redshift":waves_h["Z"],
        "galaxy center":"{}, {} [RA, DEC degrees]".format(*gal_center),
        "galaxy position angle":"{} [degrees E of N]".format(gal_pa),
        "spectral resolution":("interpolated from {} arc lamp measurements, "
                               "reported in the galaxy rest frame"
                               "".format(len(const.mitchell_arc_centers)))}
    coord_comments = {
        "target":gal_name,
        "coord-system":("dimensionless distance in plane through galaxy "
                        "center and perpendicular to line-of-sight, "
                        "expressed in arcsec - physical distance is "
                        "(given coords)*pi/(180*3600)*length_factor, "
                        "with length_factor the distance to the galaxy"),
        "distance scale factor":"line-of-sight distance to galaxy",
        "origin":"coordinate system origin at galaxy center",
        "origin":"{}, {} [RA, DEC degrees]".format(*gal_center),
        "x-direction":"East",
        "y-direction":"North",
        "galaxy major axis":"{} [degrees E of N]".format(gal_pa),
        "fiber shape":"circle",
        "fiber radius":"{} arcsec".format(const.mitchell_fiber_radius)}
    name = "{}_mitchell_datacube".format(gal_name.lower())
    ifuset = ifu.IFUspectrum(spectra=spectra[:, valid],
                             bad_data=bad_data[:, valid],
                             noise=noise[:, valid],
                             ir=spec_res_full[:, valid],
                             spectra_ids=fiber_numbers,
                             wavelengths=gal_waves[valid],
                             spectra_unit=spec_unit,
                             wavelength_unit=wave_unit,
                             comments=comments,
                             coords=cart_coords,
                             coords_unit=const.arcsec,
                             coord_comments=coord_comments,
                             linear_scale=fiber_radius,
                             footprint=fiber_circle,
                             name=name)
    ifuset.write_to_fits(output_path)
    print "  wrote proc cube: {}".format(output_path)


for plot_info in things_to_plot:
    print plot_info['data_path']
    ifuset = ifu.read_mitchell_datacube(plot_info['data_path'])

    #Collect data to plot
    nfibers = len(ifuset.coords)
    fibersize = const.mitchell_fiber_radius.value #Assuming units match!
    xcoords = -ifuset.coords[:,0] #Flipping east-west to match pictures
    ycoords = ifuset.coords[:,1]
    squaremax = np.amax(np.abs(ifuset.coords)) + fibersize
    rcoords = np.sqrt(xcoords**2 + ycoords**2)
    coordunit = ifuset.coord_comments['coordunit']
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
    pdf.close()
