"""
Script to run testing trials of pPXF
"""


import sys
sys.path.append("/stg/scripts/bin/ppxf")
sys.path.append("/stg/scripts/bin/mpfit")
import os
import re
import pickle
import warnings

import numpy as np
from scipy.interpolate import interp1d
from astropy.io import fits
from ppxf_test import ppxf_testing as ppxf

from MassiveKinematics import (speed_of_light, prepare_templates_empirical,
    compute_ir, read_miles_index, in_range_union, plot_results,
    compute_projected_confidences, prepare_templates)
import utilities as utl


def re_filesearch(pattern, directory=None):
    """
    Return a list of all files in passed directory that match the
    passed regular expression.  Only the filename, not the full
    pathnames are considering for matching, however the returned
    matches will include pathnames if a directory is specified.

    Args:
    pattern - string, regular expression pattern
    dir - string, directory name, default is current directory

    Returns: files, matches
    files - list of strings, matching file paths including directory.
    matches - list of python re match objects for each matched file.
    """
    if directory is None:
        directory = os.curdir
    files_present = os.listdir(directory)
    matching_files, match_objects = [], []
    for filename in files_present:
        match_result = re.match(pattern, filename)
        if match_result:
            matching_files.append("{}/{}".format(directory, filename))
            match_objects.append(match_result)
    return matching_files, match_objects


def read_dict_file(filename, delimiter=None, comment='#', skip=False,
                   conversions=[float, str], specific={}):
    """
    Parse a two column text file into a dictionary, with the first
    column becoming the keys and the second the values.  If a key
    appears more than once in the file, then its value in the
    dictionary will be a list of all values given in the file arranged
    by order of appearance. Values are read as floats, and then
    strings, though custom conversion can be specified.

    Args:
    filename - string
        Name of file to be parsed. The file must contain two columns
        separated by the passed delimiter. The first delimiter
        encountered on each line will mark the column division,
        subsequent delimiters are treated as part of the value data.
        Leading and trailing whitespace is stripped from each line.
        Blank lines will be skipped.
    delimiter - string, default is any whitespace
        String separating keys and values
    comment - string, default='#'
       Anything after a comment character on a line is ignored
    conversions - list, default = [float, str]
        A list of functions mapping strings to some python object, to
        be applied to each value before storing in the output dict.
        Conversions are tried in order until success.
    specific - dict, no default
        A dict mapping keys in the file to conversion functions. When
        a key appearing in the file has an entry in specific, the
        function given in specific is used first before attempting
        the functions given in conversions.
    skip - bool, default = False
        If False, a line that had no successful conversion attempts
        will throw an exception, otherwise it is skipped silently.

    Returns - file_dict
    file_dict - dictionary with all key and converted value pairs
    """
    max_splits = 1
    file_dict = {}
    number_encountered = {}
    with open(filename, 'r') as infofile:
        for line in infofile:
            comment_location = line.find(comment)  # is -1 if nothing found
            comment_present = (comment_location != -1)
            if comment_present:
                line = line[:comment_location]
            if len(line) == 0:
                continue
            label, raw_data = line.strip().split(delimiter, max_splits)
                # split(None, X) splits on any whitespace
            try:
                conv_data = specific[label](raw_data)
            except (KeyError, ValueError):
                # no specific conversion function given or conversion failed
                for conversion_func in conversions:
                    try:
                        conv_data = conversion_func(raw_data)
                    except ValueError: # conversion failed
                        pass
                    else: # conversion success
                        break
                else:
                    if not skip:
                        raise ValueError("All conversions failed.")
            if label in number_encountered:
                number_encountered[label] += 1
            else:
                number_encountered[label] = 1
            if number_encountered[label] == 1:
                file_dict[label] = conv_data
            elif number_encountered[label] == 2:
                file_dict[label] = [file_dict[label], conv_data]
            elif number_encountered[label] >= 3:
                file_dict[label].append(conv_data)
    return file_dict


def write_results_files(params, error, chisq_dof, temp_weights,
                        covmat, temp_info, results_filename,
                        title, add_poly, mult_poly):
    """ Write file summarizing ppxf fit to MASSIVE spectral bin """
    with open(results_filename, 'w') as results_file:
        results_file.write("pPXF fit results: {}\n".format(title))
        results_file.write("chi^2/dof = {:7.3f}\n".format(chisq_dof))
        results_file.write("v         = {:7.3f} +- {:7.3f} km/s\n"
                           "".format(params[0], error[0]*np.sqrt(chisq_dof)))
        results_file.write("sigma     = {:7.3f} +- {:7.3f} km/s\n"
                           "".format(params[1], error[1]*np.sqrt(chisq_dof)))
        missing = 6 - params.shape[0]
        if missing > 0:  # fit less than 6 params, need to add zeros
            params = np.concatenate((params, np.zeros(missing)))
            error = np.concatenate((error, np.zeros(missing)))
        elif missing < 0:
            raise Exception("Invalid bestfit parameter format")
        for h_num, (h_value, h_error) in enumerate(zip(params[2:], error[2:])):
            results_file.write("h{}        = {:7.3f} +- {:7.3f}\n"
                               "".format(3 + h_num, h_value,
                                         h_error*np.sqrt(chisq_dof)))
        results_file.write("covariance matrix:\n")
        for row in covmat:
            for element in row:
                results_file.write("{:6.3e}  ".format(element))
            results_file.write("\n")
        results_file.write("templates: {}/{} nonzero\n"
                          "".format(np.sum(temp_weights > 0),
                                    temp_weights.shape[0]))
        results_file.write("weight%\tmiles#\tname\t\ttype"
                          "\t\t\t\tTeff\tlogg\t[Fe/H]\n")
        sorted_template_num = np.argsort(temp_weights)[::-1]
        percentage_weights = temp_weights*100.0/np.sum(temp_weights)
        for template_num in sorted_template_num:
            if temp_weights[template_num] > 0:
                results_file.write("{:05.2f}\t{}\n"
                                   "".format(percentage_weights[template_num],
                                             temp_info[template_num]))
        results_file.write("additive polynomial degree: {}\n"
                           "".format(len(add_poly)))
        results_file.write("additive polynomial weights:\n")
        for w in add_poly:
            results_file.write("{}\n".format(w))
        results_file.write("multiplicative polynomial degree: {}\n"
                           "".format(len(mult_poly)))
        results_file.write("multiplicative polynomial weights:\n")
        for w in mult_poly:
            results_file.write("{}\n".format(w))
    return


# read ppxf and target settings
settings_filenames = sys.argv[1:]  # target.txt, kinematics.txt
range_function = lambda s: map(float, s.split("-"))
specific = {"mask_observed":range_function,
            "fit_range":lambda s: map(float, s.split("-"))}
knobs = {}
for filename in settings_filenames:
    knobs.update(read_dict_file(filename, specific=specific))
# read binned spectra
binned_spectra_paths = re_filesearch(knobs['binned_spectra_pattern'],
                                     directory=knobs['data_dir'])[0]
binned_spectra_paths.sort()  # by bin number, depends on filename format
binned_data, s2n, logscales, bin_numbers = [], [], [], []
for spectrum_path in binned_spectra_paths:
    bin_numbers.append(int(spectrum_path.split('/')[-1][3:5]))
    hdu = fits.open(spectrum_path)
    data = hdu[0].data
    hdu.close()
    spectrum, noise, wavelengths, bad_pixels = data
    valid = ~bad_pixels.astype(bool)  # bad_pixels stored as float 1.0 or 0.0
    log_delta = np.log(wavelengths[1:]/wavelengths[:-1])
    log_spaced = np.max(np.absolute(log_delta - log_delta[0])) < 10**(-10)
    if not log_spaced:
        warnings.warn("Data is not log spaced: {}\n"
                      "Interpolating...".format(spectrum_path))
        slop = 10**(-10)
        num_pixels = wavelengths.size
        start = np.min(wavelengths)*(1 + slop)
        stop = np.max(wavelengths)*(1 - slop)
        log_waves = np.exp(np.linspace(np.log(start), np.log(stop),
                                       num_pixels))
        interp_func_spec = interp1d(wavelengths, spectrum)
        interp_func_noise = interp1d(wavelengths, noise)
        spectrum = interp_func_spec(log_waves)
        noise = interp_func_noise(log_waves)
        new_mask = np.zeros(log_waves.size)
        bad_pixel_numbers = np.arange(valid.size)[~valid]
        bad_pixel_delta = np.zeros(bad_pixel_numbers.size)
        for delta_index, bad_pixel_num in enumerate(bad_pixel_numbers):
            if bad_pixel_num == 0:
                delta = (wavelengths[bad_pixel_num + 1] -
                         wavelengths[bad_pixel_num])
            elif bad_pixel_num == (valid.size - 1):
                delta = (wavelengths[bad_pixel_num] -
                         wavelengths[bad_pixel_num - 1])
            else:
                upper = (wavelengths[bad_pixel_num + 1] -
                         wavelengths[bad_pixel_num])
                lower = (wavelengths[bad_pixel_num] -
                         wavelengths[bad_pixel_num - 1])
                delta = np.max([upper, lower])
            bad_pixel_delta[delta_index] = delta
        for wave_index, w in enumerate(log_waves):
            dist_to_bad = np.absolute(w - wavelengths[~valid])
            if np.any(dist_to_bad < bad_pixel_delta):
                new_mask[wave_index] = True
        mask = new_mask.copy()
        wavelengths = log_waves.copy()
        data = np.array([spectrum, noise, wavelengths, mask])
    s2n.append(np.mean(spectrum[valid]/noise[valid]))
    binned_data.append(data)
    log_delta = np.log(wavelengths[1:]/wavelengths[:-1])
    logscales.append(log_delta[0])
binned_data = np.array(binned_data)
    # all of the relevant spectral data for each bin; for bin number
    # n: spectrum, noise, wavelengths, bad_pixels = binnned_data[n]
s2n = np.array(s2n)  # the mean s/n over valid pixels for each bin
logscales = np.array(logscales)  # uniform sampling step in log(wavelength)
pixels_indicies = np.arange(binned_data.shape[2],
                            dtype=int)  # index over wavelength sampling
# get instrument resolutions
binned_ir_paths = re_filesearch(knobs['ir_pattern'],
                                directory=knobs['data_dir'])[0]
binned_ir_paths.sort()  # by bin number, depends on filename format
binned_ir = []
for ir_path in binned_ir_paths:
    binned_ir.append(np.loadtxt(ir_path))
binned_ir = np.array(binned_ir)

# gather template library
template_spectra = {}
template_wavelengths = {}
template_index = {}
for tmp_source in ['obv', 'lib']:
    template_spectra[tmp_source] = []
    template_pattern = knobs[tmp_source + "_template_pattern"]
    template_dir = knobs[tmp_source + "_template_dir"]
    template_index_filename = knobs[tmp_source + "_template_index_filename"]

    template_paths = re_filesearch(template_pattern, template_dir)[0]
    template_paths.sort()  # by template number, 0001 to 0985
    for step, template_path in enumerate(template_paths):
        template_data = np.loadtxt(template_path)  # cols: waves, spectrum
        template_spectra[tmp_source].append(template_data[:, 1])
        if step == 0:  # only need this once - assume all others identical
            template_wavelengths[tmp_source] = template_data[:, 0]
    template_spectra[tmp_source] = np.array(template_spectra[tmp_source])
    template_index_path = "{}/{}".format(template_dir,
                                         template_index_filename)
    template_index[tmp_source] = read_miles_index(template_index_path)
    index_filename = ('{}-{}_template_index-full.txt'
                      ''.format(tmp_source, knobs["name"]))
    np.savetxt(index_filename, template_index[tmp_source],
               fmt='%s', delimiter='  ')

# proceed with fits
fit_start, fit_stop = knobs["fit_range"]
bin_wavelengths = binned_data[0, 2, :]  # assume samplings identical
out_of_bounds = ((bin_wavelengths < fit_start) |
                 (fit_stop < bin_wavelengths ))
fit_waves = bin_wavelengths[~out_of_bounds]
fit_data = binned_data[:, :, ~out_of_bounds]
masked = in_range_union(fit_waves, knobs["mask_observed"])
guess = ([knobs["v_guess"], knobs["sigma_guess"]] +
         [knobs["hn_guess"]]*int(knobs["moments_to_fit"] - 2))
bias = knobs['bias']
add_degree = int(knobs['additive_degree'])
mult_degree = int(knobs['multiplicative_degree'])
num_moments = int(knobs['moments_to_fit'])
# setup outputs
run_only_common = ["masked", "wavelengths", "noise", "bin_number"]
both_common = ["spectrum"]
both_double = ["gh_params", "tempweights", "model",
               "mult_params", "mult_model", "add_params",
               "add_model", "chisq_dof", "gh_params_errors"]
run_output = {"common":{out:[] for out in run_only_common + both_common},
                 'obv':{out:[] for out in both_double},
                 'lib':{out:[] for out in both_double}}
# prepare empirical templates
full_ir_obvw = compute_ir(template_wavelengths['obv'], binned_ir[0, ...])
obv_templates, obv_template_waves = (
    prepare_templates_empirical(template_spectra['obv'],
                                template_wavelengths['obv'],
                                full_ir_obvw, logscales[0],
                                utl.min_max(template_wavelengths['obv'])))
run_output["obv"]["templates"] = obv_templates
run_output["obv"]["templates_waves"] = obv_template_waves
# run fits
limit = 10**(-6)
run_output['common']["wavelengths"].append(fit_waves)
for bin_index, bin_number in enumerate(bin_numbers):
    print 'fitting bin {}'.format(bin_number)
    bin_output = {"common":{out:[] for out in both_common},
                     'obv':{out:[] for out in both_double},
                     'lib':{out:[] for out in both_double}}
    bin_spectrum = fit_data[bin_index, 0, :]
    bin_noise = fit_data[bin_index, 1, :]
    bin_badpixels = fit_data[bin_index, 3, :].astype(bool) | masked
    good_pixels_indicies = np.nonzero(~bin_badpixels)[0]
    run_output['common']["bin_number"].append(bin_number)
    run_output['common']["spectrum"].append(bin_spectrum)
    run_output['common']["noise"].append(bin_noise)
    run_output['common']["masked"].append(bin_badpixels)
    # empirical fit
    log_obvtemplate_initial = np.log(obv_template_waves.min())
    log_galaxy_initial = np.log(fit_waves.min())
    velocity_offset_obv = (log_obvtemplate_initial -
                       log_galaxy_initial)*speed_of_light
    velscale = logscales[bin_index]*speed_of_light
    obv_results = ppxf(obv_templates.T,  # templates in cols
                       bin_spectrum, bin_noise, velscale, guess,
                       goodpixels=good_pixels_indicies, bias=bias,
                       moments=num_moments, degree=add_degree,
                       vsyst=velocity_offset_obv, mdegree=mult_degree,
                       plot=False, quiet=True)
    obv_v, obv_sigma = obv_results.sol[:2]
    gaussian_converged = False  # initialize
    matching_ir = binned_ir.copy()
    print '...shifting and smoothing'
    iters = 0
    while ~gaussian_converged:
        bin_ir = compute_ir(template_wavelengths['lib'],
                            matching_ir[bin_index])
        bin_output["lib"]["ir"] = bin_ir
        lib_templates, lib_template_waves = (
            prepare_templates(template_spectra['lib'],
                              template_wavelengths['lib'],
                              knobs["lib_template_fwhm"], bin_ir,
                              logscales[bin_index],
                              utl.min_max(template_wavelengths['lib'])))
        log_libtemplate_initial = np.log(lib_template_waves.min())
        velocity_offset_lib = (log_libtemplate_initial -
                               log_galaxy_initial)*speed_of_light
        lib_results = ppxf(lib_templates.T,  # templates in cols
                           bin_spectrum, bin_noise, velscale,
                           obv_results.sol,
                           goodpixels=good_pixels_indicies, bias=bias,
                           moments=num_moments, degree=add_degree,
                           vsyst=velocity_offset_lib, mdegree=mult_degree,
                           plot=False, quiet=True)
        bin_output["lib"]["templates"] = lib_templates
        bin_output["lib"]["templates_waves"] = lib_template_waves
        lib_v, lib_sigma = lib_results.sol[:2]
        delta_v = obv_v - lib_v
        v_converged = np.absolute(delta_v) < np.absolute(obv_v)*limit
        delta_sigma = obv_sigma - lib_sigma
        sigma_converged = (np.absolute(delta_sigma) <
                           np.absolute(obv_sigma)*limit)
        gaussian_converged = v_converged & sigma_converged
        delta_lambda = matching_ir[bin_index,:,0]*delta_sigma/speed_of_light
        delta_fwhm = delta_lambda*2*np.sqrt(2*np.log(2))
        matching_ir[bin_index, :, 1] -= delta_fwhm
        template_wavelengths['lib'] /= (1 + delta_v/speed_of_light)
        iters += 1
    print "converged: {} iterations".format(iters)
    print "monte carlo..."
    fit_results = {'obv':obv_results, 'lib':lib_results}
    preped_templates = {'obv':obv_templates, 'lib':lib_templates}
    v_offsets = {'obv':velocity_offset_obv, 'lib':velocity_offset_lib}
    for tmp_source, results in fit_results.iteritems():
        bestfit_model = results.bestfit
        chisq_dof = results.chi2
        try:
            bestfit_addpoly_weights = results.polyweights
            add_continuum = (
                np.polynomial.legendre.legval(
                    np.linspace(-1, 1, fit_waves.shape[0]),
                    bestfit_addpoly_weights))
        except AttributeError:
            bestfit_addpoly_weights = np.array([])
            add_continuum = np.array([])
        try:
            bestfit_multpoly_weights = results.mpolyweights
            mult_continuum = (
                np.polynomial.legendre.legval(
                    np.linspace(-1, 1, fit_waves.shape[0]),
                    np.concatenate(([1], bestfit_multpoly_weights))))
        except AttributeError:
            bestfit_multpoly_weights = np.array([])
            mult_continuum = np.array([])
        # save results
        run_output[tmp_source]["gh_params"].append(results.sol)
        run_output[tmp_source]["chisq_dof"].append(chisq_dof)
        run_output[tmp_source]["model"].append(bestfit_model)
        run_output[tmp_source]["tempweights"].append(results.weights)
        run_output[tmp_source]["mult_params"].append(bestfit_multpoly_weights)
        run_output[tmp_source]["mult_model"].append(mult_continuum)
        run_output[tmp_source]["add_params"].append(bestfit_addpoly_weights)
        run_output[tmp_source]["add_model"].append(add_continuum)
    # error simulation
    num_sims = int(knobs["error_simulation_trials"])
    if num_sims < 10:
        for tmp_source, results in fit_results.iteritems():
            scaled_error = results.error*np.sqrt(results.chi2)
            run_output[tmp_source]["gh_params_errors"].append(scaled_error)
            bin_output[tmp_source]["covmat"] = np.nan*np.ones((num_moments,
                                                               num_moments))
    else:
        for tmp_source, results in fit_results.iteritems():
            sample_fits = []
            for trial in xrange(num_sims):
                noise_draw = np.random.randn(*bin_spectrum.shape)
                    # uniform, uncorrelated Gaussian noise
                simulated_galaxy = bin_spectrum + noise_draw*bin_noise
                bin_output['common']["spectrum"].append(simulated_galaxy)
                sample_fitter = ppxf(preped_templates[tmp_source].T,  # templates in cols
                                     simulated_galaxy, bin_noise, velscale,
                                     guess, goodpixels=good_pixels_indicies,
                                     moments=num_moments, degree=add_degree,
                                     vsyst=v_offsets[tmp_source], mdegree=mult_degree,
                                     bias=0.0,  # bias always 0 for noise sims
                                     plot=False, quiet=True)
                sample_fits.append(sample_fitter)
                try:
                    sample_addpoly_weights = sample_fitter.polyweights
                    sample_add_continuum = (
                        np.polynomial.legendre.legval(
                            np.linspace(-1, 1, fit_waves.shape[0]),
                            sample_addpoly_weights))
                except AttributeError:
                    sample_addpoly_weights = np.array([])
                    sample_add_continuum = np.array([])
                try:
                    sample_multpoly_weights = sample_fitter.mpolyweights
                    sample_mult_continuum = (
                        np.polynomial.legendre.legval(
                            np.linspace(-1, 1, fit_waves.shape[0]),
                            np.concatenate(([1], sample_multpoly_weights))))
                except AttributeError:
                    sample_multpoly_weights = np.array([])
                    sample_mult_continuum = np.array([])
                bin_output[tmp_source]["gh_params"].append(sample_fitter.sol)
                bin_output[tmp_source]["chisq_dof"].append(sample_fitter.chi2)
                bin_output[tmp_source]["model"].append(sample_fitter.bestfit)
                bin_output[tmp_source]["tempweights"].append(sample_fitter.weights)
                bin_output[tmp_source]["mult_params"].append(sample_multpoly_weights)
                bin_output[tmp_source]["mult_model"].append(sample_mult_continuum)
                bin_output[tmp_source]["add_params"].append(sample_addpoly_weights)
                bin_output[tmp_source]["add_model"].append(sample_add_continuum)
            covmat, sim_paramerror = (
                compute_projected_confidences(np.asarray(bin_output[tmp_source]["gh_params"])))
            run_output[tmp_source]["gh_params_errors"].append(sim_paramerror)
            bin_output[tmp_source]["covmat"] = covmat
    for name, dictionary in bin_output.iteritems():
        for subname, data in dictionary.iteritems():
            bin_output[name][subname] = np.asarray(data)
    # write output
    for tmp_source in fit_results:
        fit_title = ("{}-bin{:02d}-{}"
                     "".format(knobs['name'], bin_number, tmp_source))
        for output_name, data in bin_output[tmp_source].iteritems():
            output_filename = "{}-{}.txt".format(fit_title, output_name)
            np.savetxt(output_filename, np.asarray(data).T, delimiter='  ')
        bestfit_filename =("{}-bin{:02d}-{}_{}-bestfit.p"
            "".format(knobs['name'], bin_number, fit_start, fit_stop))
for name, dictionary in run_output.iteritems():
    for subname, data in dictionary.iteritems():
        run_output[name][subname] = np.asarray(data)
for tmp_source in fit_results.keys() + ['common']:
    for output_name, data in run_output[tmp_source].iteritems():
        output_filename = ("{}-{}-{}.txt"
                           "".format(knobs['name'], tmp_source, output_name))
        np.savetxt(output_filename, np.asarray(data).T, delimiter='  ')