"""
Script to run testing trials of pPXF
"""


import sys
sys.path.append("/stg/scripts/bin/ppxf")
sys.path.append("/stg/scripts/bin/mpfit")
import os
import re
import pickle

import numpy as np
from astropy.io import fits
from ppxf_test import ppxf_testing as ppxf

from MassiveKinematics import (speed_of_light, prepare_templates,
    compute_ir, read_miles_index, in_range_union, plot_results,
    compute_projected_confidences)


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
    binned_data.append(data)
    spectrum, noise, wavelengths, bad_pixels = data
    valid = ~bad_pixels.astype(bool)  # bad_pixels stored as float 1.0 or 0.0
    s2n.append(np.mean(spectrum[valid]/noise[valid]))
    log_delta = np.log(wavelengths[1:]/wavelengths[:-1])
    log_spaced = np.max(np.absolute(log_delta - log_delta[0])) < 10**(-10)
    if not log_spaced:
        raise Exception("Data is not log spaced!")
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
template_paths = re_filesearch(knobs['template_pattern'],
                               directory=knobs["template_dir"])[0]
template_paths.sort()  # by template number, 0001 to 0985
template_spectra = []
for step, template_path in enumerate(template_paths):
    template_data = np.loadtxt(template_path)  # col 0 waves, col 1 spectrum
    template_spectra.append(template_data[:, 1])
    if step == 0:  # only need this once - assume all others identical
        template_wavelengths = template_data[:, 0]
template_spectra = np.array(template_spectra)
template_range = [template_wavelengths.min(), template_wavelengths.max()]
template_index_path = "{}/{}".format(knobs["template_dir"],
                                     knobs["template_index_filename"])
template_index = read_miles_index(template_index_path)
index_filename = '{}-template_index-full.txt'.format(knobs["name"])
np.savetxt(index_filename, template_index, fmt='%s', delimiter='  ')

# fit spectra
if type(knobs["fit_range"][0]) == float:
    knobs["fit_range"] = [knobs["fit_range"]]
for fit_start, fit_stop in knobs["fit_range"]:
    bin_wavelengths = binned_data[0, 2, :]  # assume samplings identical
    out_of_bounds = ((bin_wavelengths < fit_start) |
                   (fit_stop < bin_wavelengths ))
    masked = in_range_union(bin_wavelengths, knobs["mask_observed"])
    masked = masked | out_of_bounds
    guess = ([knobs["v_guess"], knobs["sigma_guess"]] +
             [knobs["hn_guess"]]*int(knobs["moments_to_fit"] - 2))
    bias = knobs['bias']
    add_degree = int(knobs['additive_degree'])
    mult_degree = int(knobs['multiplicative_degree'])
    num_moments = int(knobs['moments_to_fit'])
    run_only = ["masked", "wavelengths", "noise",
                "bin_number", "gh_params_errors"]
    run_and_bin = ["gh_params", "tempweights", "model",
                   "mult_params", "mult_model", "add_params",
                   "add_model", "chisq_dof", "spectrum"]
    bin_only = ["prepared_templates"]
    run_output = {out:[] for out in run_only + run_and_bin}
    # get templates
    fortemps_index = 0
    print ("determining {}_{} templates from bin{:02d}"
           "".format(fit_start, fit_stop, bin_numbers[fortemps_index]))
    fortemps_spectrum = binned_data[fortemps_index, 0, :]
    fortemps_noise = binned_data[fortemps_index, 1, :]
    fortemps_badpixels = binned_data[fortemps_index, 3, :].astype(bool)
    fortemps_ir = binned_ir[fortemps_index, ...]
    bad_pixels = masked | fortemps_badpixels
    good_pixels_indicies = np.nonzero(~bad_pixels)[0]
    interpolated_ir = compute_ir(template_wavelengths, fortemps_ir)
    prepared_templates, prepared_template_waves = (
        prepare_templates(template_spectra, template_wavelengths,
                          knobs["template_fwhm"], interpolated_ir,
                          logscales[fortemps_index], template_range))
    log_template_initial = np.log(prepared_template_waves.min())
    log_galaxy_initial = np.log(bin_wavelengths.min())
    velocity_offset = (log_template_initial -
                       log_galaxy_initial)*speed_of_light
    velscale = logscales[fortemps_index]*speed_of_light
    ppxf_fitter = ppxf(prepared_templates.T,  # templates in cols
                       fortemps_spectrum, fortemps_noise, velscale,
                       guess, goodpixels=good_pixels_indicies, bias=bias,
                       moments=num_moments, degree=add_degree,
                       vsyst=velocity_offset, mdegree=mult_degree,
                       plot=False, quiet=True)
    unnormed_weights = ppxf_fitter.weights
    normed_weights = unnormed_weights/np.sum(unnormed_weights)
    nonzero = normed_weights > 10**(-10)
    subindex = [entry for entry, is_nonzero in
                zip(template_index, nonzero) if is_nonzero]
    subindex_filename = ('{}-template_index-{}_{}.txt'
                         ''.format(knobs["name"], fit_start, fit_stop))
    np.savetxt(subindex_filename, subindex, fmt='%s', delimiter='  ')
    selected_template_spectra = template_spectra[nonzero, :]
    print 'fitting with {}/{} templates'.format(
        selected_template_spectra.shape[0], nonzero.size)
    # run fits
    for bin_index, bin_number in enumerate(bin_numbers):
        print ("fitting {}_{} bin{:02d}"
               "".format(fit_start, fit_stop, bin_number))
        bin_output = {}
        bin_spectrum = binned_data[bin_index, 0, :]
        bin_noise = binned_data[bin_index, 1, :]
        bin_badpixels = binned_data[bin_index, 3, :].astype(bool)
        bin_ir = binned_ir[bin_index, ...]
        bad_pixels = masked | bin_badpixels
        good_pixels_indicies = np.nonzero(~bad_pixels)[0]
        run_output["wavelengths"].append(bin_wavelengths)
        run_output["bin_number"].append(bin_number)
        run_output["spectrum"].append(bin_spectrum)
        run_output["noise"].append(bin_noise)
        run_output["masked"].append(bad_pixels)
        # template prep
        interpolated_ir = compute_ir(template_wavelengths, bin_ir)
        prepared_templates, prepared_template_waves = (
            prepare_templates(selected_template_spectra,
                              template_wavelengths, knobs["template_fwhm"],
                              interpolated_ir, logscales[bin_index],
                              template_range))
        bin_output["templates"] = prepared_templates
        bin_output["templates_waves"] = prepared_template_waves
        # actual fit
        log_template_initial = np.log(prepared_template_waves.min())
        log_galaxy_initial = np.log(bin_wavelengths.min())
        velocity_offset = (log_template_initial -
                           log_galaxy_initial)*speed_of_light
        velscale = logscales[bin_index]*speed_of_light
        ppxf_fitter = ppxf(prepared_templates.T,  # templates in cols
                           bin_spectrum, bin_noise, velscale, guess,
                           goodpixels=good_pixels_indicies, bias=bias,
                           moments=num_moments, degree=add_degree,
                           vsyst=velocity_offset, mdegree=mult_degree,
                           plot=False, quiet=True)
        bestfit_model = ppxf_fitter.bestfit
        chisq_dof = ppxf_fitter.chi2
        try:
            bestfit_addpoly_weights = ppxf_fitter.polyweights
            add_continuum = (
                np.polynomial.legendre.legval(
                    np.linspace(-1, 1, wavelengths.shape[0]),
                    bestfit_addpoly_weights))
        except AttributeError:
            bestfit_addpoly_weights = np.array([])
            add_continuum = np.array([])
        try:
            bestfit_multpoly_weights = ppxf_fitter.mpolyweights
            mult_continuum = (
                np.polynomial.legendre.legval(
                    np.linspace(-1, 1, wavelengths.shape[0]),
                    np.concatenate(([1], bestfit_multpoly_weights))))
        except AttributeError:
            bestfit_multpoly_weights = np.array([])
            mult_continuum = np.array([])
        # save results
        run_output["gh_params"].append(ppxf_fitter.sol)
        run_output["chisq_dof"].append(chisq_dof)
        run_output["model"].append(bestfit_model)
        run_output["tempweights"].append(ppxf_fitter.weights)
        run_output["mult_params"].append(bestfit_multpoly_weights)
        run_output["mult_model"].append(mult_continuum)
        run_output["add_params"].append(bestfit_addpoly_weights)
        run_output["add_model"].append(add_continuum)
        # error simulation
        num_sims = int(knobs["error_simulation_trials"])
        if num_sims > 1:
            for output_name in run_and_bin:
                last_addition = np.array(run_output[output_name][-1])
                samples_shape = [num_sims] + list(last_addition.shape)
                bin_output[output_name] = np.zeros(samples_shape)
            sample_fits = []
            for trial in xrange(num_sims):
                noise_draw = np.random.randn(*bin_spectrum.shape)
                    # uniform, uncorrelated Gaussian noise
                simulated_galaxy = bin_spectrum + noise_draw*bin_noise
                bin_output["spectrum"][trial, ...] = simulated_galaxy
                sample_fitter = ppxf(prepared_templates.T,  # templates in cols
                                     simulated_galaxy, bin_noise, velscale,
                                     guess, goodpixels=good_pixels_indicies,
                                     moments=num_moments, degree=add_degree,
                                     vsyst=velocity_offset, mdegree=mult_degree,
                                     bias=0.0,  # bias always 0 for noise sims
                                     plot=False, quiet=True)
                sample_fits.append(sample_fitter)
                try:
                    sample_addpoly_weights = sample_fitter.polyweights
                    sample_add_continuum = (
                        np.polynomial.legendre.legval(
                            np.linspace(-1, 1, wavelengths.shape[0]),
                            sample_addpoly_weights))
                except AttributeError:
                    sample_addpoly_weights = np.array([])
                    sample_add_continuum = np.array([])
                try:
                    sample_multpoly_weights = sample_fitter.mpolyweights
                    sample_mult_continuum = (
                        np.polynomial.legendre.legval(
                            np.linspace(-1, 1, wavelengths.shape[0]),
                            np.concatenate(([1], sample_multpoly_weights))))
                except AttributeError:
                    sample_multpoly_weights = np.array([])
                    sample_mult_continuum = np.array([])
                bin_output["gh_params"][trial, ...] = sample_fitter.sol
                bin_output["chisq_dof"][trial, ...] = sample_fitter.chi2
                bin_output["model"][trial, ...] = sample_fitter.bestfit
                bin_output["tempweights"][trial, ...] = sample_fitter.weights
                bin_output["mult_params"][trial, ...] = sample_multpoly_weights
                bin_output["mult_model"][trial, ...] = sample_mult_continuum
                bin_output["add_params"][trial, ...] = sample_addpoly_weights
                bin_output["add_model"][trial, ...] = sample_add_continuum
            covmat, sim_paramerror = (
                compute_projected_confidences(bin_output["gh_params"]))
            run_output["gh_params_errors"].append(sim_paramerror)
            sample_fits_filename =("{}-bin{:02d}-{}_{}-mcfits.p"
                "".format(knobs['name'], bin_number, fit_start, fit_stop))
            # with open(sample_fits_filename, 'wb') as sample_fits_file:
            #     pickle.dump(sample_fits, sample_fits_file)
        else:
            sim_paramerror = ppxf_fitter.error*np.sqrt(chisq_dof)
        # write output
        fit_title = ("{}-bin{:02d}-{}_{}"
                     "".format(knobs['name'], bin_number,
                               fit_start, fit_stop))
        for output_name, data in bin_output.iteritems():
            output_filename = "{}-{}.txt".format(fit_title, output_name)
            np.savetxt(output_filename, np.asarray(data).T, delimiter='  ')
        bestfit_filename =("{}-bin{:02d}-{}_{}-bestfit.p"
            "".format(knobs['name'], bin_number, fit_start, fit_stop))
        # with open(bestfit_filename, 'wb') as bestfit_file:
        #     pickle.dump(ppxf_fitter, bestfit_file)
        plot_results(bin_spectrum, bestfit_model, bin_noise, bin_wavelengths,
                     bad_pixels, bestfit_addpoly_weights,
                     bestfit_multpoly_weights, chisq_dof, fit_title,
                     "{}.png".format(fit_title))
        write_results_files(ppxf_fitter.sol, sim_paramerror, chisq_dof,
                            ppxf_fitter.weights, covmat, template_index,
                            "{}-results.txt".format(fit_title), fit_title,
                            bestfit_addpoly_weights,
                            bestfit_multpoly_weights)
    for output_name, data in run_output.iteritems():
        output_filename = ("{}-{}_{}-{}.txt"
                           "".format(knobs['name'], fit_start,
                                     fit_stop, output_name))
        np.savetxt(output_filename, np.asarray(data).T, delimiter='  ')