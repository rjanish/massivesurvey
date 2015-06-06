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
bins_to_fit  = [1, 4, 31, 49]
# bins_to_fit = 1 + np.arange(56)  # all bins
setup_settings = { # from old settings files
    "name":                     "testrgitun",
    "template_lib_name":        'miles-massive',
    "templates_to_use":         const.fullMILES_1600fullgalaxy_optimized,
    "mask":                     [[4260.0, 4285.0],
                                 [4775.0, 4795.0]],  # A
    "error_simulation_trials":  0}
gh_init = [0, 250, 0, 0, 0, 0]
fit_range = [3900.0, 5300.0]
range_masks = [3900, 5300]
fit_settings = {"add_deg":     0,
                "mul_deg":     7,
                "num_moments": 6,
                "bias":        0.0}  # A
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
        raise ValueError("Invalid datacube path {}, "
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
    specset_to_fit = specset.get_subset(bins_to_fit)
    # do fits
    driver = driveppxf.pPXFDriver(spectra=specset_to_fit,
                                  templates=template_library,
                                  fit_range=fit_range,
                                  initial_gh=gh_init,
                                  **fit_settings)
    results = driver.run_fit()