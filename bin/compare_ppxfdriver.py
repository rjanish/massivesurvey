"""
This script validates the output of massive.pPXFdriver against the
older scripts ppxf_trials.py and MassiveKinematics.py

output:

"""


import os
import re
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

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
                    help="directory in which to place the fit comparisons")
parser.add_argument("--to_fit", action="store", type=int, nargs='*',
                    help="bin numbers of the bins to fit")
parser.add_argument("--fit_all", action="store_true",
                    help="fit all bins")
args = parser.parse_args()
old_ppxf_dir = args.comparison_dir
dest_dir = args.destination_dir

# read old ppxf settings and data filenames
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
gauss_guess = [settings["v_guess"], settings["sigma_guess"]]
hn_guess = [settings["hn_guess"]]*(int(settings["moments_to_fit"]) - 2)
old_gh_initial = gauss_guess + hn_guess
old_misc_settings = {"bias": settings["bias"],
                     "mul_deg": settings["multiplicative_degree"],
                     "add_deg": settings["additive_degree"],
                     "num_moments": settings["moments_to_fit"]}
nominal_fit_range = map(float, settings["fit_range"].split('-'))
masked_intervals = [map(float, interval.split('-'))
                    for interval in settings["mask_observed"]]
old_data_dir = os.path.join(old_ppxf_dir, settings["data_dir"])
old_binneddata_paths = utl.re_filesearch(settings["binned_spectra_pattern"],
                                                  old_data_dir)[0]
old_ir_paths = utl.re_filesearch(settings["ir_pattern"], old_data_dir)[0]
old_temp_dir = os.path.join(old_ppxf_dir, settings["template_dir"])
old_temp_paths = utl.re_filesearch(settings["template_pattern"],
                                   old_temp_dir)[0]

# gather old results
old_results_dir = os.path.join(old_ppxf_dir, 'results')
param_pattern = r'{}.*gh_params.txt'.format(settings['name'])
all_gh_params_paths = utl.re_filesearch(param_pattern, old_results_dir)[0]
full_params_path = [path for path in all_gh_params_paths
                    if 'bin' not in path][0]
error_pattern = r'{}.*gh_params_errors.txt'.format(settings['name'])
all_gh_errors_paths = utl.re_filesearch(error_pattern, old_results_dir)[0]
full_errors_path = [path for path in all_gh_errors_paths
                    if 'bin' not in path][0]
temps_pattern = r'{}.*?bin\d{{2}}.*templates.txt'.format(settings['name'])
temps_paths = utl.re_filesearch(temps_pattern, old_results_dir)[0]
temps_waves_pattern = (r'{}.*?bin\d{{2}}.*templates_waves.txt'
                       r''.format(settings['name']))
temps_waves_paths = utl.re_filesearch(temps_waves_pattern, old_results_dir)[0]
print 'found old results files:'
print "  {}".format(os.path.basename(full_params_path))
print "  {}".format(os.path.basename(full_errors_path))
print ("  {} ppxf-output templates, e.g.\n    {}"
       "".format(len(temps_paths), os.path.basename(temps_paths[0])))
print ("  {} ppxf-output template wavelengths, e.g.\n    {}"
       "".format(len(temps_waves_paths),
                 os.path.basename(temps_waves_paths[0])))
old_params = np.loadtxt(full_params_path).T
old_errors = np.loadtxt(full_errors_path).T

# get target bins
num_old_bins = old_params.shape[0] - 1  # do not count full-galaxy bin
if args.fit_all:
    to_fit = np.arange(1, 1 + num_old_bins, dtype=int)
else:
    to_fit = np.asarray(args.to_fit, dtype=int)
print "reading bins:"
print "  {}".format(to_fit)

# gather old data
spectra, noise = [], []
bad_data_cropping, bad_data_masking = [], []
    # these two bad data arrays implement the two methods of handling the
    # fit_range enforcement - later cropping or masking the edges
templates, template_waves = [], []
ir_samples, ir_interped = [], []
ids = []
zipped_old_paths = zip(old_binneddata_paths, temps_paths,
                       temps_waves_paths, old_ir_paths)
for binned_path, temp_path, tempw_path, ir_path in zipped_old_paths:
    current_ids = []
    for path in [binned_path, temp_path, tempw_path]:
        id = int(re.search("bin(\d{2})", os.path.basename(path)).groups()[0])
        current_ids.append(id)
    current_ids = np.asarray(current_ids, dtype=int)
    if np.all(current_ids == current_ids[0]):
        ids.append(current_ids[0])
    else:
        raise ValueError("data and templates bin id numbers do not agree")
    [[s, n, w, m]], headers = utl.fits_quickread(binned_path)
    waves = w  # assume uniform wavelength sampling
    spectra.append(s)
    noise.append(n)
    masked = utl.in_union_of_intervals(waves, masked_intervals)
    out_of_range = ~utl.in_linear_interval(waves, nominal_fit_range)
    bad_data_cropping.append(m.astype(bool) | masked)
    bad_data_masking.append(m.astype(bool) | masked | out_of_range)
    ir_samples.append(np.loadtxt(ir_path)) # col 0: waves, col 1: fwhm
    obsolete_interp = utl.interp1d_constextrap(*ir_samples[-1].T)
    ir_interped.append(obsolete_interp(waves))
    templates.append(np.loadtxt(temp_path).T)
    template_waves.append(np.loadtxt(tempw_path))
ids = np.asarray(ids, dtype=int)
spectra = np.asarray(spectra)
noise = np.asarray(noise)
bad_data_cropping = np.asarray(bad_data_cropping, dtype=bool)
bad_data_masking = np.asarray(bad_data_masking, dtype=bool)
waves = np.asarray(waves)
    # These are wavelengths in the galaxy's rest frame, obtained by applying
    # some assumed redshift to the observed wavelengths
ir_samples = np.asarray(ir_samples)
ir_interped = np.asarray(ir_interped)

# make specset
duplicate_name="{}-duplicate".format(settings["name"])
duplicate_specset = spec.SpectrumSet(spectra=spectra, noise=noise,
                                        wavelengths=waves,
                                        bad_data=bad_data_masking,
                                        ir=ir_interped,
                                        spectra_ids=ids,
                                        spectra_unit=const.flux_per_angstrom,
                                        wavelength_unit=const.angstrom,
                                        name=duplicate_name)
duplicate_specset = duplicate_specset.get_subset(to_fit)

# make template library for each bin
template_libs = {}
for id, temps_s, temp_w, ir_samps in zip(ids, templates,
                                        template_waves, ir_samples):
    obsolete_interp = utl.interp1d_constextrap(*ir_samps.T)
    ir_interped_to_temps = obsolete_interp(temp_w)
        # here interpolate the ir onto the galaxy wavelengths which are
        # in the galaxy rest frame, but use an interpolation function
        # defined by measurements in the instrument rest frame
    temps_ir = np.asarray([ir_interped_to_temps,]*temps_s.shape[0])
    temp_ids = np.arange(temps_s.shape[0])  # ID LAzINESS
    lib = temps.TemplateLibrary(spectra=temps_s,
                                bad_data=np.zeros(temps_s.shape, dtype=bool),
                                noise=np.zeros(temps_s.shape, dtype=float),
                                ir=temps_ir,
                                spectra_ids=temp_ids, wavelengths=temp_w,
                                spectra_unit=const.flux_per_angstrom,
                                wavelength_unit=const.angstrom,
                                comments={"source":"output of old ppxf fits"},
                                name="old bin {} fit templates".format(id),
                                catalog=np.zeros(temps_s.shape[0]))
    template_libs[id] = lib

# run fits
for bin_id in to_fit:
    print "fitting {}".format(bin_id)
    scale_inward = 1 + np.asarray([1, -1])*const.float_tol
    full_range = duplicate_specset.spec_region*scale_inward
    driver_duplicate = driveppxf.pPXFDriver(
                            specset=duplicate_specset.get_subset([bin_id]),
                            templib=template_libs[bin_id],
                            fit_range=full_range,
                            initial_gh=old_gh_initial,
                            num_trials=0,
                            **old_misc_settings)
    driver_duplicate.run_fit()

sys.exit()

# plot comparison
for param in xrange(6):
    fig, ax = plt.subplots()
    ax.errorbar(to_fit, old_params[to_fit, param],
                yerr=old_errors[to_fit, param],
                linestyle='', marker='o', alpha=0.3, color='k',
                label='old results')
    ax.plot(to_fit, oldir_masked_results[:, param],
                linestyle='', marker='.', alpha=0.8, color='r',
                label=specset_oldir_masked.name)
    ax.plot(to_fit, oldir_cropped_results[:, param],
                linestyle='', marker='.', alpha=0.8, color='b',
                label=specset_oldir_cropped.name)
    ax.legend(loc='best')
    ax.set_xlabel("bin number")
    ax.set_ylabel("gh param {}".format(param))
    ax.set_title("fit comparisons - {}".format(param))
    fig.savefig("ppxfdriver_compare-ngc1600-{}.pdf".format(param))