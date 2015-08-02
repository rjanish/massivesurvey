"""
Test the velocity shifting of raw datacubes.
"""


import argparse
import re
import os

import numpy as np
import pandas as pd
import shapely.geometry as geo

import utilities as utl
import massivepy.constants as const
import massivepy.spectralresolution as res
import massivepy.spectrum as spec
import massivepy.io as mpio

# get cmd line arguments
parser = argparse.ArgumentParser(description=__doc__,
                formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("paramfiles", nargs='*', type=str,
                    help="path(s) to input parameter file(s)")
args = parser.parse_args()
all_paramfile_paths = args.paramfiles

for paramfile_path in all_paramfile_paths:
    # parse input parameter file
    input_params = utl.read_dict_file(paramfile_path)
    raw_cube_path = input_params['raw_mitchell_cube']
    target_positions = pd.read_csv(input_params["target_positions"],
                                   comment='#', sep="[ \t]+",
                                   engine='python')
    destination_dir = input_params['destination_dir']
    if not os.path.isdir(destination_dir):
        raise ValueError("Invalid destination dir {}".format(destination_dir))
    output_filename = lambda gal_name: "{}_mitchellcube.fits".format(gal_name)

    # start processing
    # check galaxy name consistency
    ngc_match = re.search(mpio.re_gals['NGC'], raw_cube_path)
    if ngc_match is None:
        msg = "No galaxy name found for path {}".format(raw_cube_path)
        raise RuntimeError(msg)
    else:
        ngc_num = ngc_match.groups()[0]
    data, headers = utl.fits_quickread(raw_cube_path)
    ngcs = [re.search(mpio.re_gals['NGC'], header["OBJECT"]).groups()[0]
            for header in headers]
    all_match = [num == ngc_num for num in ngcs] == [True]*len(ngcs)
    if not all_match:
        raise RuntimeError("Datacube headers do not match galaxy name "
                           " 'NGC {}' found in path".format(ngc_num))
    # TODO: add unit checking against header
    spec_unit = const.flux_per_angstrom  # assume spectrum units
    wave_unit = const.angstrom  # assume wavelength units
    fiber_radius = const.mitchell_fiber_radius.value  # arcsec
    fiber_circle = lambda center: geo.Point(center).buffer(fiber_radius)
    # get data
    ngc_name = "NGC{}".format(ngc_num)
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
    print "  interpolating spectral resolution..."
    specres_inst = np.nan*np.ones(spectra.shape)
    specres_gal = np.nan*np.ones(spectra.shape)
    for fiber_iter, fiber_res_samples in enumerate(spec_res_samples):
        inst_interp = utl.interp1d_constextrap(*fiber_res_samples.T)
        specres_inst[fiber_iter] = inst_interp(gal_waves)
            # This ignores the differences between ir frames
        galframe_samples = fiber_res_samples/(1 + redshift)
        gal_interp = utl.interp1d_constextrap(*galframe_samples.T)
        specres_gal[fiber_iter] = gal_interp(gal_waves)
            # This uses scales the ir correctly into the galaxy rest frame
    print ("cropping to {}-{} A (galaxy rest frame)..."
           "".format(*const.mitchell_crop_region))
    valid = utl.in_linear_interval(gal_waves, const.mitchell_crop_region)
    # TO DO: actual masking of sky lines, check for nonsense values
    bad_data = np.zeros(spectra.shape, dtype=bool)
    # save data
    fiber_numbers = np.arange(spectra.shape[0], dtype=int)
    vhelio = spectra_h["VHELIO"]
    general_comments = {
        "target":ngc_name,
        "heliocentric correction applied":"{} [km/s]".format(vhelio),
        "wavelengths":"wavelength in galaxy rest frame",
        "applied galaxy redshift":waves_h["Z"],
        "galaxy center":"{}, {} [RA, DEC degrees]".format(*gal_center),
        "galaxy position angle":"{} [degrees E of N]".format(gal_pa),
        # needs more detail
        "spectral resolution":("interpolated from {} arc lamp "
            "measurements".format(len(const.mitchell_arc_centers)))}
    name = "{}_mitchell_datacube".format(ngc_name.lower())
    ifuset = sepc.SpectrumSet(spectra=spectra[:, valid],
                              bad_data=bad_data[:, valid],
                              noise=noise[:, valid],
                              ir=spec_res_full[:, valid],
                              spectra_ids=fiber_numbers,
                              wavelengths=gal_waves[valid],
                              spectra_unit=spec_unit,
                              wavelength_unit=wave_unit,
                              comments=comments, name=name)
    # output_path = os.path.join(destination_dir, output_filename(ngc_name))
    # ifuset.write_to_fits(output_path)
    # print "  wrote proc cube: {}".format(output_path)
