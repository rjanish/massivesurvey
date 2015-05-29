"""
This is a wrapper for ppxf, facilitating kinematic fitting.
"""


import warnings
import functools

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
    def __init__(self, spectra=None, templates=None, fit_settings=None):
        """
        Args:
        spectra - SpectraSet object
            The spectra which will be fit by pPXF. Must be sampled
            with logarithmic spacing and have units of flux/velocity.
        templates - TemplateLibrary object
        fit_settings - dictionary of fit settings
        """
        self.spectra = spectra
        self.templates = templates
        self.fit_settings = dict(fit_settings)

    def ready_to_fit(self):
        """ all inputs ready for ppxf """
        logscales = {}
        for name, specset in zip(["spectra", "templates"],
                                 [self.spectra, self.templates.spectrumset]):
            if specset.is_log_sampled():
                logscale = np.log(specset.waves[1]/specset.waves[0])
                logscales[name] = logscale
            else:
                warnings.warn("{} are not log-spaced".format(name))
                return False, "{}-spacing".format(name)
        delta_logscale = (logscales["spectra"] -
                          logscales["templates"])/logscales["spectra"]
        logscale_matches = np.absolute(delta_logscale) < const.float_tol
        if not logscale_matches:
            warnings.warn("log-scales of spectra and templates do not match")
            return False, "templates-spacing"
        fitrange = self.fit_settings["fit_range"]
        delta_lambda = fitrange[1] - fitrange[0]
        spec_flux = self.spectra.compute_flux(interval=fitrange)
        temp_flux = self.templates.spectrumset.compute_flux(interval=fitrange)
        for name, flux_set in zip(["spectra", "templates"],
                                  [spec_flux, temp_flux]):
            flux_delta = (flux_set - delta_lambda)/delta_lambda
            flux_matches = np.all(np.absolute(flux_delta) < const.float_tol)
            if not flux_matches:
                warnings.warn("{} are not properly normalized".format(name))
                return False, "{}-normalization".format(name)
        # TO DO: settings_needed
        return True, None

    def prepare_fit(self):
        """
        """
        try:
            self.spectra.log_resample()
        except ValueError, msg:
            print "skipping spectra log_resample - {}".format(msg)
        logscale = np.log(self.spectra.waves[1]/self.spectra.waves[0])
        try:
            self.templates.spectrumset.log_resample(logscale=logscale)
        except ValueError, msg:
            print "skipping template log_resample - {}".format(msg)
        to_fit = utl.in_linear_interval(self.spectra.waves,
                                        self.fit_settings["fit_range"])
        target_flux = (self.fit_settings["fit_range"][1] -
                       self.fit_settings["fit_range"][0])
        fitregion_flux = functools.partial(spec.SpectrumSet.compute_flux,
                                    interval=self.fit_settings["fit_range"])
        fitregion_flux.__name__ = "compute_flux_of_fitting_region"
        self.spectra = self.spectra.get_normalized(norm_func=fitregion_flux,
                                                   norm_value=target_flux)
        self.templates.spectrumset.metaspectra["bad_data"] = (
            self.templates.spectrumset.metaspectra["bad_data"].astype(bool))
        # HACK - this should not need to be here, FIX THIS
        self.templates.spectrumset = (
            self.templates.spectrumset.get_normalized(
                norm_func=fitregion_flux, norm_value=target_flux))
        return

    def run_fit(self):
        """
        """
        # self.prepare_fit() - this function is likely evil
        # if not self.ready_to_fit(): - possibly this one too
        #     warnings.warn("fit setup invalid, aborting")
        #     return
        add_deg = int(self.fit_settings["additive_degree"])
        mul_deg = int(self.fit_settings["multiplicative_degree"])
        num_moments = int(self.fit_settings["moments_to_fit"])
        bias = float(self.fit_settings["bias"])
        v_guess = float(self.fit_settings["v_guess"])
        sigma_guess = float(self.fit_settings["sigma_guess"])
        hn_guess = float(self.fit_settings["hn_guess"])
        fit_range = np.asarray(self.fit_settings["fit_range"])
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

