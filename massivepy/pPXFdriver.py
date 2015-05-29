"""
This is a wrapper for ppxf, facilitating kinematic fitting.
"""


import warnings
import functools
import copy

import numpy as np
import ppxf

import utilities as utl
import massivepy.constants as const
import massivepy.spectrum as spec


class pPXFDriver(object):
    """
    This class is a driver for sets of pPXF fits, aiding in particular
    the preparation of spectra and the recording of results.

    Before allowing a pPXF fit to occur, this class will ensure that
    both the observed spectra and template spectra are log-sampled in
    wavelength and have identical spectral resolutions.

    All of the data and settings input to the fit are stored as class
    attributes. Intermediate computations and fit results are
    similarly stored. A pickle of the entire class will serve as an
    exact record of the fit, and I/O methods are provided to simplify
    recording of individual results.
    """
    REQUIRED_SETTINGS = ["additive_degree", "multiplicative_degree",
                         "moments_to_fit", "bias", "v_guess", "sigma_guess",
                         "hn_guess", "fit_range"]

    def __init__(self, spectra=None, templates=None, **kwargs):
        """
        Args:
        spectra - SpectraSet object
            The spectra which will be fit by pPXF. Must be sampled
            with logarithmic spacing and have units of flux/velocity.
        templates - TemplateLibrary object
        settings - dictionary of fit settings
        """
        self.spectra = copy.deepcopy(spectra)
        self.templates = copy.deepcopy(templates)
        if "settings" in kwargs:
            self.settings = dict(settings)
        else:
            self.setting = dict(kwargs)
        for setting in self.REQUIRED_SETTINGS:
            if setting not in self.settings:
                raise ValueError("{} must be specified to fit spectra"
                                 "".format(setting))
        # prepare copied spectra
        try:
            self.spectra.log_resample()
        except ValueError, msg:
            print "skipping spectra log_resample ({})".format(msg)
        self.to_fit = utl.in_linear_interval(self.spectra.waves,
                                             self.settings["fit_range"])
        self.target_flux = (self.settings["fit_range"][1] -
                            self.settings["fit_range"][0])
            # this sets numerical spectral values near 1
        self.get_flux = functools.partial(spec.SpectrumSet.compute_flux,
                                          interval=self.settings["fit_range"])
            # this computes the flux of a SpectrumSet only within the
            # fitting range - useful for normalization of templates
        self.get_flux.__name__ = "compute_flux_of_fitting_region"
        self.spectra = self.spectra.get_normalized(norm_func=fitregion_flux,
                                                   norm_value=target_flux)


    def run_fit(self):
        """
        """
        add_deg = int(self.settings["additive_degree"])
        mul_deg = int(self.settings["multiplicative_degree"])
        num_moments = int(self.settings["moments_to_fit"])
        bias = float(self.settings["bias"])
        v_guess = float(self.settings["v_guess"])
        sigma_guess = float(self.settings["sigma_guess"])
        hn_guess = float(self.settings["hn_guess"])
        fit_range = np.asarray(self.settings["fit_range"])
        guess = ([v_guess, sigma_guess] +
                 [hn_guess]*int(num_moments - 2))
        results = np.zeros((self.spectra.num_spectra, num_moments + 2))
        for bin_iter, bin_num in enumerate(self.spectra.ids):
            print "fitting bin {}".format(bin_num)
            single_spec = self.spectra.get_subset([bin_num])
            print "cropping to {}".format(fit_range)
            single_spec = single_spec.crop(fit_range)
            print single_spec.waves.min(), single_spec.waves.max()
            # construct matched-resolution library
            spec_ir = self.spectra.metaspectra["ir"][0, :]
            spec_waves = self.spectra.waves
            spec_ir_inerp_func = utl.interp1d_constextrap(spec_waves, spec_ir)
                # use to get ir(template_waves); this will require some
                # extrapolation beyond the spectra wavelength range, which
                # the above function will do as a constant
            template_waves = self.templates.spectrumset.waves
            single_ir_to_match = spec_ir_inerp_func(template_waves)
            num_temps = self.templates.spectrumset.num_spectra
            ir_to_match = np.asarray([single_ir_to_match,]*num_temps)
            matched_library = self.templates.match_resolution(ir_to_match)
            # resample
            logscale = np.log(single_spec.waves[1]/single_spec.waves[0])
            matched_library.spectrumset.log_resample(logscale)
            # norm
            [spec_flux] = single_spec.compute_flux()
            fitregion_flux = functools.partial(spec.SpectrumSet.compute_flux,
                                               interval=fit_range)
            fitregion_flux.__name__ = "compute_flux_of_fitting_region"
            matched_library.spectrumset.metaspectra["bad_data"] = (
                matched_library.spectrumset.metaspectra["bad_data"].astype(bool))
            # HACK - this should not need to be here, FIX THIS
            matched_library.spectrumset = matched_library.spectrumset.get_normalized(
                norm_func=fitregion_flux, norm_value=spec_flux)

              # DEBUG
            test_output = "results-ppxf/matched_lib_bin{:02d}.fits".format(bin_num)
            matched_library.spectrumset.write_to_fits(test_output)
            print "lib status test:"
            print "spec flux", single_spec.compute_flux()
            print "tlib flux", matched_library.spectrumset.compute_flux(interval=fit_range)
            print "spec samples", single_spec.is_log_sampled(), np.log(single_spec.waves[1]/single_spec.waves[0])
            print "tlib samples", matched_library.spectrumset.is_log_sampled(), np.log(matched_library.spectrumset.waves[1]/matched_library.spectrumset.waves[0])
            print "spec ir", single_ir_to_match
            print "tlib ir", matched_library.spectrumset.metaspectra["ir"][0, :]
            print "tlib ir std", matched_library.spectrumset.metaspectra["ir"].std(axis=0).max()
              # DEBUG

            # do fit
            log_temp_initial = np.log(matched_library.spectrumset.waves.min())
            log_spec_initial = np.log(single_spec.waves.min())
            velocity_offset = (log_temp_initial - log_spec_initial)*const.c_kms
            velscale = logscale*const.c_kms
            good_pixels_indicies = np.nonzero(~single_spec.metaspectra["bad_data"][0])[0]
            ppxf_fitter = ppxf.ppxf(matched_library.spectrumset.spectra.T, # templates in cols
                               single_spec.spectra[0],
                               single_spec.metaspectra["noise"][0],
                               velscale, guess,
                               goodpixels=good_pixels_indicies,
                               bias=bias, moments=num_moments,
                               degree=add_deg, mdegree=mul_deg,
                               vsyst=velocity_offset, plot=False, quiet=True)
            results[bin_iter, 0] = bin_num
            results[bin_iter, 1] = ppxf_fitter.chi2
            results[bin_iter, 2:] = ppxf_fitter.sol
        return np.asarray(results)

