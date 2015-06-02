"""
This script validates the output of massive.pPXFdriver against the
older scripts ppxf_trials.py and MassiveKinematics.py

output:

"""


import os
import re
import argparse
import sys # DEVELOPMENT

import numpy as np

import utilities as utl
import massivepy.constants as const
import massivepy.templates as temps
import massivepy.spectrum as spec
import massivepy.pPXFdriver as driveppxf


# get cmd line arguments
parser = argparse.ArgumentParser(description=__doc__,
                formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("comparison_dir",
                    help="The directory containing an old (ppxf_trials.py, "
                         "MassiveKinematics.py) pPXF fit by which to compare "
                         "the fit produced by massivpy.pPXFdriver. This must "
                         "be a directory containing 'kinematics' and "
                         "'target' fit settings files as well as a 'results' "
                         "directory containing fit results")
parser.add_argument("--destination_dir", action="store",
                    help="Directory in which to place the fit comparisons")
args = parser.parse_args()
old_ppxf_dir = args.comparison_dir
dest_dir = args.destination_dir

print "reading old pPXF fit: {}".format(old_ppxf_dir)
old_settings_paths = utl.re_filesearch(".*?(kinematics|target).*?\.txt",
                                       old_ppxf_dir)[0]
print "found {} settings files:".format(len(old_settings_paths))
settings = {}
for settings_path in old_settings_paths:
    print "  {}".format(os.path.basename(settings_path))
    settings.update(utl.read_dict_file(settings_path))
print "old settings:"
for setting, value in settings.iteritems():
    print "  {}: {}".format(setting, value)
# read old fit files
old_data_dir = os.path.join(old_ppxf_dir, settings["data_dir"])
old_spectra = utl.re_filesearch(settings["binned_spectra_pattern"],
                                old_data_dir)
old_ir = utl.re_filesearch(settings["ir_pattern"], old_data_dir)
old_temp_dir = os.path.join(old_ppxf_dir, settings["template_dir"])
old_templates = utl.re_filesearch(settings["template_pattern"],
                                  old_temp_dir)[0]
old_temp_index = os.path.join(old_temp_dir,
                              settings["template_index_filename"])
old_temp_ir = float(settings["template_fwhm"])
# read old fit settings
gauss_guess = [settings["v_guess"], settings["sigma_guess"]]
hn_guess = [settings["hn_guess"]]*(int(settings["moments_to_fit"]) - 2)
old_gh_initial = gauss_guess + hn_guess
old_misc_settings = {"bias": settings["bias"],
                     "mul_deg": settings["multiplicative_degree"],
                     "add_deg": settings["additive_degree"],
                     "num_moments": settings["moments_to_fit"]}
old_fit_range = map(float, settings["fit_range"].split('-'))
masked_intervals = [map(float, interval.split('-'))
                    for interval in settings["mask_observed"]]



sys.exit()

# # FIT SETTINGS

# setup_settings = {
#     "template_lib_name":        'miles-massive',
#     "templates_to_use":         const.fullMILES_1600fullgalaxy_optimized,
#     "mask":                     [[4260.0, 4285.0],
#                                  [4775.0, 4795.0]],  # A
#     "error_simulation_trials":  0}
# gh_init = [0, 250, 0, 0, 0, 0]
# fit_range = [3900.0, 5300.0]
# # test range enforced by masking
# # spec already matches exactly
# range_masks = [3900, 5300]
# fit_settings = {"add_deg":     0,
#                 "mul_deg":     7,
#                 "num_moments": 6,
#                 "bias":        0.0}  # A

# uploaded_spectra_dir = "bin-spectra-check-20150522/to_upload_20150522/data"
# spec_files = utl.re_filesearch("bin\d{2}_ngc1600.fits", uploaded_spectra_dir)[0]
# spec_files.sort()
# spec_files = spec_files[1:] # drop full galaxy
# ir_files = utl.re_filesearch("bin\d{2}_ngc1600_ir.txt", uploaded_spectra_dir)[0]
# ir_files.sort()
# ir_files = ir_files[1:] # drop full galaxy

# # defaults
# datamap = utl.read_dict_file(const.path_to_datamap)
# binned_dir = datamap["binned_mitchell"]
# results_dir = "results-ppxf/"

# # get data
# spectra, noise, waves, mask, ids, irs, test_ir = [], [], [], [], [], [], []
# for path_spec, path_ir in zip(spec_files, ir_files):
#     [[s, n, w, b]], headers = utl.fits_quickread(path_spec)
#     spectra.append(s)
#     noise.append(n)
#     waves = w
#     masked = utl.in_union_of_intervals(waves, setup_settings["mask"])
#     mask_lowrange = waves < range_masks[0]
#     mask_highrange = waves > range_masks[1]
#     mask.append(b.astype(bool) | masked | mask_lowrange | mask_highrange)
#     id = int(re.search("bin(\d{2})", path_spec).groups()[0])
#     ids.append(id)
#     test_ir.append(np.loadtxt(path_ir))
# spectra = np.asarray(spectra)
# noise = np.asarray(noise)
# waves = np.asarray(waves)
# mask = np.asarray(mask, dtype=bool)
# ids = np.asarray(ids)
# test_ir = np.asarray(test_ir)
# specset = spec.SpectrumSet(spectra=spectra, bad_data=mask, noise=noise,
#                  ir=4.5*np.ones(spectra.shape), spectra_ids=ids,
#                  wavelengths=waves, spectra_unit=const.flux_per_angstrom,
#                  wavelength_unit=const.angstrom, name="testrun",
#                  test_ir=test_ir)
# specset = specset.get_subset((to_fit))


# # get templates
# temps_dir = datamap["template_libraries"]
# temp_path = os.path.join(temps_dir, setup_settings["template_lib_name"])
# full_template_library = temps.read_miles_library(temp_path)
# template_library = full_template_library.get_subset(setup_settings["templates_to_use"])


# # do fits
# driver = driveppxf.pPXFDriver(spectra=specset,
#                               templates=template_library, # overwrite this in ppxf_driver
#                               fit_range=fit_range,
#                               initial_gh=gh_init,
#                               **fit_settings)
# results, ids = driver.run_fit()
# results_path = "results-ppxf/new1600fits-test-oldlibs.txt"
# ids_path = "results-ppxf/new1600fits-ids-test-oldlibs.txt"
# np.savetxt(results_path, results)
# np.savetxt(ids_path, ids)