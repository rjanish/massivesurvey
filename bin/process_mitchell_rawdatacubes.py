"""
Process the raw Mitchell datacubes into a format more accessible
for binning and fitting.

The changes are:
 - Coordinates converted from (RA, DEC) to projected Cartesian arcsec
 - Arc frames are fit by Gaussian line profiles and replaced with the
   resulting samples of fwhm(lambda) for each fiber

output:
    one processed data for each input raw datacube
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
import massivepy.IFUspectrum as ifu


# defaults
datamap = utl.read_dict_file(const.path_to_datamap)
raw_cube_dir = datamap["raw_mitchell_cubes"]
proc_cube_dir = datamap["proc_mitchell_cubes"]
target_positions = pd.read_csv(datamap["target_positions"],
                               comment='#', sep="[ \t]+",
                               engine='python')
output_filename = lambda gal_name: "{}_mitchellcube.fits".format(gal_name)

# get cmd line arguments
parser = argparse.ArgumentParser(description=__doc__,
                formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("cubes", nargs='*', type=str,
                    help="The raw Michell datacubes to process, passed as "
                         "either a path to the cube or a galaxy name. Paths "
                         "are matched first in the current directory and "
                         "then in the MASSIVE Mitchell raw datacube dir, "
                         "while for passed galaxy names matching cubes are "
                         "searched for only in the raw datacube directory.")
parser.add_argument("--all", action="store_true",
                    help="Process all raw Michell datacubes located in "
                         "the MASSIVE Mitchell raw datacube directory")
parser.add_argument("--destination_dir", action="store",
                    type=str, nargs=1, default=proc_cube_dir,
                    help="Directory in which to place processed datacubes")
args = parser.parse_args()
if args.all:
    cube_paths = utl.re_filesearch(r".*\.fits", raw_cube_dir)[0]
else:
    cube_paths = [os.path.normpath(p) for p in args.cubes]
for path in cube_paths:
    if (not os.path.isfile(path)) or (os.path.splitext(path)[-1] != ".fits"):
        raise ValueError("Invalid raw datacube path {}, "
                         "must be .fits file".format(path))
dest_dir = os.path.normpath(args.destination_dir)
if not os.path.isdir(dest_dir):
    raise ValueError("Invalid destination dir {}".format(dest_dir))

# start processing
print "processing {} cubes ...".format(len(cube_paths))
for path in cube_paths:
    # check galaxy name consistency
    ngc_match = re.search(const.re_ngc, path)
    if ngc_match is None:
        raise RuntimeError("No galaxy name found for path {}".format(path))
    else:
        ngc_num = ngc_match.groups()[0]
    data, headers = utl.fits_quickread(path)
    ngcs = [re.search(const.re_ngc, header["OBJECT"]).groups()[0]
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
    gal_position = target_positions[target_positions.Name == ngc_name]
    gal_center = gal_position.Ra.iat[0], gal_position.Dec.iat[0]
    gal_pa = gal_position.PA_best.iat[0]
        # .ita[0] extracts scalar value from a 1-element dataframe
    print "\n{}".format(ngc_name)
    print "  raw datacube: {}".format(path)
    print "        center: {}, {}".format(*gal_center)
    print "            pa: {}".format(gal_pa)
    try:
        # wavelengths of arc spectra are specifically included
        spectra, noise, all_waves, coords, arcs, all_inst_waves = data
        spectra_h, noise_h, waves_h, coords_h, arcs_h, inst_waves_h = headers
        gal_waves = all_waves[0, :]  # assume uniform samples; gal rest frame
        inst_waves = all_inst_waves[0, :]  # instrument rest frame
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
        res_interp_func = utl.interp1d_constextrap(*fiber_res_samples.T)
        spec_res_full[fiber_iter] = res_interp_func(inst_waves)
            # spec_res_full is fwhm(galaxy-rest-frame wavelengths),
            # given as samples at the wavelength values in gal_waves
            # OR equivalently,
            # spec_res_full is fwhm(inst-rest-frame wavelengths),
            # given as samples at the wavelength values in inst_waves
    print ("cropping to {}-{} A (galaxy rest frame)..."
           "".format(*const.mitchell_crop_region))
    valid = utl.in_linear_interval(gal_waves, const.mitchell_crop_region)
    # TO DO: actual masking of sky lines, check for nonsense values
    bad_data = np.zeros(spectra.shape, dtype=bool)
    # save data
    fiber_numbers = np.arange(spectra.shape[0], dtype=int)
    vhelio = spectra_h["VHELIO"]
    comments = {
        "target":ngc_name,
        "heliocentric correction applied":"{} [km/s]".format(vhelio),
        "wavelengths":"wavelength in galaxy rest frame",
        "applied galaxy redshift":waves_h["Z"],
        "galaxy center":"{}, {} [RA, DEC degrees]".format(*gal_center),
        "galaxy position angle":"{} [degrees E of N]".format(gal_pa),
        "spectral resolution":("interpolated from {} arc lamp "
            "measurements".format(len(const.mitchell_arc_centers)))}
    coord_comments = {
        "target":ngc_name,
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
    name = "{}_mitchell_datacube".format(ngc_name.lower())
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
    output_path = os.path.join(proc_cube_dir, output_filename(ngc_name))
    ifuset.write_to_fits(output_path)
    print "  wrote proc cube: {}".format(path)
