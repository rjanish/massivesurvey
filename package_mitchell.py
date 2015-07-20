"""
This script packages loose Mitchell spectra, ir, and binning info
into a format usable by the current MASSVIE pPXF pipeline.
"""


import argparse
import os
import pickle

import numpy as np
import scipy.interpolate as interp
import pandas as pd

import utilities as utl
import massivepy.constants as const
import massivepy.spectrum as spec


# get params
parser = argparse.ArgumentParser(description=__doc__,
                formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("paramfile", type=str,
                    help="path to input parameter file for this script")
args = parser.parse_args()
input_params = utl.read_dict_file(args.paramfile)

target = input_params["target"]                 
positions_filepath = input_params["positions_filepath"]     
spec_dir = input_params["spec_dir"]               
spec_patt = input_params["spec_patt"]              
ir_dir = input_params["ir_dir"]                 
ir_patt = input_params["ir_patt"]                
if type(input_params["bad_regions"]) != list:
    input_params["bad_regions"] = [input_params["bad_regions"]]
bad_regions = np.asarray([interval.split('-') for interval
                          in input_params["bad_regions"]], dtype=float)     
output_dir = input_params["output_dir"]             
fullgal_output_filename = input_params["fullgal_output_filename"]
bins_output_filename = input_params["bins_output_filename"]        
bininfo_output_filename = input_params["bininfo_output_filename"]  
try:
    pickled_bins = True
    bin_pickle = input_params["bin_pickle"]
except:  
    pickled_bins = False
    bin_defs_path = input_params["bin_defs_path"]          
    binning_dir = input_params["binning_dir"]            
    bin_coords_patt = input_params["bin_coords_patt"]

fullgal_output_path = os.path.join(output_dir, fullgal_output_filename)
bins_output_path = os.path.join(output_dir, bins_output_filename)
bininfo_output_path = os.path.join(output_dir, bininfo_output_filename)

# get target data
target_positions = pd.read_csv(positions_filepath, comment='#',
                               sep="[ \t]+", engine='python')
gal_position = target_positions[target_positions.Name == target]
gal_center = gal_position.Ra.iat[0], gal_position.Dec.iat[0]
gal_pa = gal_position.PA_best.iat[0]
# read spectral data
spec_paths = utl.re_filesearch(spec_patt, spec_dir)[0]
ir_paths = utl.re_filesearch(ir_patt, ir_dir)[0]
print 'matching files...'
spectra, noises, all_waves, all_bad_data, irs = [], [], [], [], []
bin_ids = np.arange(len(spec_paths), dtype=int)
last = None # ensures print on first read-in
for bin_id, spec_path, ir_path in zip(bin_ids, spec_paths, ir_paths):
    print ("bin{:02d}: {} {}"
           "".format(bin_id, *map(os.path.basename, [spec_path, ir_path])))
    [spec_data], [spec_header] = utl.fits_quickread(spec_path)
    spectrum = spec_data[0, :]
    noise = spec_data[1, :]
    try:
        waves = spec_data[2, :]
        bad_data = np.asarray(spec_data[3, :], dtype=bool)
        rest_frame = "galaxy"
        wave_location = ".fits data"
        current = '4-row MASSIVE'
    except IndexError: # 2 and/or 3 data rows do not exist
        waves, junk = utl.fits_getcoordarray(spec_header)
        bad_data = np.zeros(spectrum.shape, dtype=bool)
        rest_frame = "instrument"
        wave_location = ".fits header"
        current = '2-row pipe1|pipe2'
    if last != current: # print only on first read-in or change of format 
        print "data in {} format".format(current)
        print "reading waves from {}".format(wave_location)
        print "assuming wavelengths in {} rest frame".format(rest_frame)
    last = current
    ir = np.loadtxt(ir_path)
    spectra.append(spectrum)
    noises.append(noise)
    all_bad_data.append(bad_data)
    all_waves.append(waves)
    irs.append(ir)
spectra = np.asarray(spectra, dtype=float)
noises = np.asarray(noises, dtype=float)
all_waves = np.asarray(all_waves, dtype=float)
all_bad_data = np.asarray(all_bad_data, dtype=bool)
irs = np.asarray(irs, dtype=float)
# re-structure
wave_mismatch = all_waves.std(axis=0)/np.mean(all_waves, axis=0)
if np.max(wave_mismatch) < const.float_tol:
    common_waves = all_waves[0]
else:
    raise Exception("wavelengths do not match!")
# find bad data
common_bad_data = utl.in_union_of_intervals(waves, bad_regions)
all_bad_data = all_bad_data | common_bad_data
interped_ir = np.zeros(spectra.shape)
for spec_iter, ir_samples in enumerate(irs):
    centers, fwhm = ir_samples.T
    fwhm_func = utl.interp1d_constextrap(centers, fwhm)
    interped_ir[spec_iter, :] = fwhm_func(common_waves)
# make .fits file
basecomments = [
    ("target", "{}".format(target)),
    ("heliocentric correction applied", " -- [km/s]"),
        # this is burried in some pefsm.fits file
    ("wavelengths", "wavelengths in {} rest frame".format(rest_frame)),
    ("galaxy center", "{}, {} [RA, DEC degrees]".format(*gal_center)),
    ("galaxy position angle", "{} [degrees E of N]".format(gal_pa)),
    ("spectral resolution", ("interpolated from {} arc lamp measurements, "
                             "reported in the {} rest frame"
                             "".format(rest_frame,
                                       len(const.mitchell_arc_centers))))]
contents_fullgal = [("contents",
                     "full galaxy spectrum (coaddition of all fibers)")]
contents_spacialbins = [("contents", "spacial IFU bins")]
specset_fullgal = spec.SpectrumSet(spectra=spectra[:1],
                                   bad_data=all_bad_data[:1],
                                   noise=noises[:1], ir=interped_ir[:1],
                                   spectra_ids=bin_ids[:1],
                                   wavelengths=common_waves,
                                   wavelength_unit=const.angstrom,
                                   spectra_unit=const.flux_per_angstrom,
                                   comments=dict(basecomments +
                                                 contents_fullgal),
                                   name="{}_fullgalaxy".format(target))
specset_fullgal.write_to_fits(fullgal_output_path)
print "wrote full galaxy to {}".format(fullgal_output_path)
specset_bins = spec.SpectrumSet(spectra=spectra[1:],
                                bad_data=all_bad_data[1:],
                                noise=noises[1:], ir=interped_ir[1:],
                                spectra_ids=bin_ids[1:],
                                wavelengths=common_waves,
                                wavelength_unit=const.angstrom,
                                spectra_unit=const.flux_per_angstrom,
                                comments=dict(basecomments +
                                              contents_spacialbins),
                                name="{}_bins".format(target))
specset_bins.write_to_fits(bins_output_path)
print "wrote spacial bins to {}".format(bins_output_path)

# get bin data
if pickled_bins:
    print 'reading binning pickle'
    with open(bin_pickle, 'rb') as pkle:
        bin_data = pickle.load(pkle)
    [fibers_in_bin, annuli, bin_cart_centers,
     bin_flux, fiber_flux, fiber_coords] = bin_data
    num_bins = len(fibers_in_bin)
    bin_defs = []
    for fibers, poly in fibers_in_bin:
        if (len(fibers) == 1) and (poly is None):
            bin_defs.append(np.asarray([[np.nan]*4]))
    for annulus in annuli:
        r_bounds = np.asarray(annulus[0])
        theta_bounds = np.asarray(annulus[1]) 
            # this is only from ma to ma, need the reflected piece as well
        
        last_theta = theta_bounds[-1, -1]
        reflected_theta_bounds = (2*last_theta - theta_bounds)[::-1, ::-1]
        theta_bounds = np.concatenate((theta_bounds, reflected_theta_bounds))
        theta_bounds = theta_bounds % 360  # shift into [0, 360)
        theta_bounds = np.deg2rad(theta_bounds)
       
        r_bounds_stacked = np.vstack([r_bounds]*theta_bounds.shape[0])
        bin_bounds = np.hstack([r_bounds_stacked, theta_bounds])
        bin_defs.append(bin_bounds)
    bin_defs = np.concatenate(bin_defs)


else:
    print 'reading binning txt files'
    bin_defs = np.genfromtxt(bin_defs_path)
    bin_defs = bin_defs[1:, 1:]
        # 1st row is full galaxy bin, drop this
        # 1st col is a 'p', identifing the bins as polar format
    angle_ma = (90.0 - gal_pa)
    bin_defs[:, 2:] += angle_ma
    bin_defs[:, 2:] = bin_defs[:, 2:] % 360  # shift into [0, 360)
    bin_defs[:, 2:] = np.deg2rad(bin_defs[:, 2:])
        # bin angels are given in bin_defs.list in degrees from side of the
        # major axis closest to east, incresing towards north. This converts
        # to angles in radians from east towards north in radians
    bin_coord_paths = utl.re_filesearch(bin_coords_patt, binning_dir)[0]
    bin_coord_paths.sort()
    bin_coord_paths = bin_coord_paths[1:] # drop full-galaxy bin
    num_bins = bin_defs.shape[0]
    bin_coords = np.zeros((num_bins, 4))
    num_fibers = np.zeros(num_bins)
    for bin_iter, path in enumerate(bin_coord_paths):
        fiber_data = pd.read_csv(path, engine='python', sep='\s{3,}')
            # delimiter of >= 3 or whitespaces as some col names have spaces
        fluxes = np.asarray(fiber_data["Med.Flux"])
        x = np.asarray(fiber_data["dRA"])  # ra arcsecs from center of gal
        y = np.asarray(fiber_data["dDec"])  # dec arcsecs from center of gal
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        total_flux = fluxes.sum()
        flux_avg = lambda x: np.sum(x*fluxes)/total_flux
        bin_coords[bin_iter, 0] = flux_avg(x)
        bin_coords[bin_iter, 1] = flux_avg(y)
        bin_coords[bin_iter, 2] = flux_avg(r)
        bin_coords[bin_iter, 3] = flux_avg(theta)
        fiber_numbers = np.asarray(fiber_data['F #'])
        num_fibers[bin_iter] = np.unique(fiber_numbers).size

    datatype = {'names':['binid','nfibers','x','y','r','th',
                         'rmin','rmax','thmin','thmax'],
                 'formats':2*['i4']+8*['f32']}
    bininfo = np.zeros(num_bins, dtype=datatype)
    bininfo['binid'] = np.arange(1, 1 + num_bins)
    bininfo['nfibers'] = num_fibers
    bininfo['x'] = bin_coords[:, 0]
    bininfo['y'] = bin_coords[:, 1]
    bininfo['r'] = bin_coords[:, 2]
    bininfo['th'] = bin_coords[:, 3]
    bininfo['rmin'] = bin_defs[:, 0]
    bininfo['rmax'] = bin_defs[:, 1]
    bininfo['thmin'] = bin_defs[:, 2]
    bininfo['thmax'] = bin_defs[:, 3]
    np.savetxt(bininfo_output_path, bininfo, delimiter='\t',
               fmt=2*['%1i']+8*['%9.5f'],
               header=' '.join(datatype['names']))
    print "wrote bin info to {}".format(bininfo_output_path)
