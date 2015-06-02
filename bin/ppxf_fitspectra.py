"""
This is a temporary testing script!

This script accepts spectral datacubes and fits each spectrum therein
using pPXF. The fit settings are currently hardcoded in the first few
lines.

output:
    One file per input spectra datacube, contaminating the best-fit
    Gauss-Hermite kinematic parameters of the input spectra.
"""

import os
import re
import argparse

import numpy as np

import utilities as utl
import massivepy.constants as const
import massivepy.templates as temps
import massivepy.spectrum as spec
import massivepy.pPXFdriver as driveppxf



# FIT SETTINGS
setup_settings = { # from old settings files
    "name":                     "testrun",
    "template_lib_name":        'miles-massive',
    "templates_to_use":         const.fullMILES_1600fullgalaxy_optimized,
    "mask":                     [[4260.0, 4285.0],
                                 [4775.0, 4795.0]],  # A
    "error_simulation_trials":  0}
fit_settings = { # from old settings files
    "additive_degree":          0,
    "multiplicative_degree":    7,
    "moments_to_fit":           6,
    "bias":                     0.0,
    "v_guess":                  0,
    "sigma_guess":              250,
    "hn_guess":                 0,
    "fit_range":                [3900.0, 5300.0]}  # A

# defaults
datamap = utl.read_dict_file(const.path_to_datamap)
binned_dir = datamap["binned_mitchell"]

# get cmd line arguments
parser = argparse.ArgumentParser(description=__doc__,
                formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("bin_filebase", nargs='*', type=str,
                    help="The filename suffix indicating the set of "
                         "binned spectra to fit, e.g. passing 'ngc5557' "
                         "will fit the datacube ngc5557.fits. Files are "
                         "searched for in the binned Mitchell directory.")
parser.add_argument("--destination_dir", action="store",
                    help="Directory in which to place the fit results.")
args = parser.parse_args()
binned_specsets_paths = [os.path.join(binned_dir, "{}.fits".format(p))
                         for p in args.bin_filebase]
binned_specsets_paths = map(os.path.normpath, binned_specsets_paths)
for path in binned_specsets_paths:
    if not os.path.isfile(path):
        raise ValueError("Invalid raw datacube path {}, "
                         "must be .fits file".format(path))
dest_dir = os.path.normpath(args.destination_dir)
if not os.path.isdir(dest_dir):
    raise ValueError("Invalid destination dir {}".format(dest_dir))

# get templates
temps_dir = datamap["template_libraries"]
temp_path = os.path.join(temps_dir, setup_settings["template_lib_name"])
full_template_library = temps.read_miles_library(temp_path)
template_library = full_template_library.get_subset(setup_settings["templates_to_use"])
# set masking
for path in binned_specsets_paths:
    specset = spec.read_datacube(path)
    masked = utl.in_union_of_intervals(specset.waves, setup_settings["mask"])
    for spec_iter in xrange(specset.num_spectra):
        specset.metaspectra["bad_data"][spec_iter, :] = (
            specset.metaspectra["bad_data"][spec_iter, :] | masked)

    # do fits
    driver = driveppxf.pPXFDriver(spectra=specset,
                                  templates=template_library,
                                  fit_settings=fit_settings)
    results = driver.run_fit()
    output_path = os.path.join(dest_dir, "ppxf_output_full.txt")
    np.savetxt(output_path, results)