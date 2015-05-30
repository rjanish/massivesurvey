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
    PPXF_REQUIRED_INPUTS = ["add_deg":int, "mul_deg":int,
                            "num_moments":int, "bias":float,]
        # These are input parameters and types needed by pPXF's caller

    def __init__(self, spectra=None, templates=None,
                 fit_range=None, initial_gh=None, **kwargs):
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
        self.fit_range = np.asarray(fit_range, dtype=float)
        if self.fit_range.shape != (2,):
            rasie ValueError("Invalid fit range shape {}"
                             "".format(self.fit_range.shape))
        if "settings" in kwargs:
            self.settings = dict(settings)
        else:
            self.setting = dict(kwargs)
        for setting in self.REQUIRED_SETTINGS:
            if setting not in self.settings:
                raise ValueError("{} must be specified to fit spectra"
                                 "".format(setting))
        self.settings = [setting:PPXF_REQUIRED_INPUTS[setting](value)
                         for setting, value in self.settings.iteritems()]
        self.initial_gh = np.asarray(initial_gh, dtype=float)
        if self.initial_gh.shape != (self.settings["num_moments"],):
            rasie ValueError("Invalid starting gh parameters shape {}, "
                             "must match number of moments to fit {}"
                             "".format(self.initial_gh.shape,
                                       self.settings["num_moments"]))
        # prepare copied spectra
        try:
            self.spectra.log_resample()
        except ValueError, msg:
            print "skipping spectra log_resample ({})".format(msg)
        self.to_fit = utl.in_linear_interval(self.spectra.waves,
                                             self.fit_range)
        self.target_flux = self.fit_range[1] - self.fit_range[0]
            # normalizing the flux to equal the numerical wavelength
            # range sets the numerical spectral values near 1
        self.get_flux = functools.partial(spec.SpectrumSet.compute_flux,
                                          interval=self.fit_range)
            # this computes the flux of a SpectrumSet only within the
            # fitting range - useful for normalization of templates
        self.get_flux.__name__ = "compute_flux_of_fitting_region"
        self.spectra = self.spectra.get_normalized(norm_func=fitregion_flux,
                                                   norm_value=target_flux)




    def run_fit(self):
        """
        """
        for spec_iter, spec_id in enumerate(self.spectra.ids):
            target_spec = self.spectra.get_subset([bin_num])
            target_spec = target_spec.crop(self.fit_range)
            exact_fit_range = utl.min_max(target_spec.waves)

            matched_library = self.prepare_library(target_spec)

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
            logscale = np.log(target_spec.waves[1]/target_spec.waves[0])
            matched_library.spectrumset.log_resample(logscale)
            # norm
            [spec_flux] = target_spec.compute_flux()
            fitregion_flux = functools.partial(spec.SpectrumSet.compute_flux,
                                               interval=fit_range)
            fitregion_flux.__name__ = "compute_flux_of_fitting_region"
            matched_library.spectrumset.metaspectra["bad_data"] = (
                matched_library.spectrumset.metaspectra["bad_data"].astype(bool))
            # HACK - this should not need to be here, FIX THIS
            matched_library.spectrumset = matched_library.spectrumset.get_normalized(
                norm_func=fitregion_flux, norm_value=spec_flux)

            # do fit
            template_range = utl.min_max(matched_library.spectrumset.waves)
            log_temp_initial = np.log(template_range[0])
            log_spec_initial = np.log(exact_fit_range[0])
            velocity_offset = (log_temp_initial - log_spec_initial)*const.c_kms
            velscale = logscale*const.c_kms
            good_pixels_indicies = np.nonzero(~target_spec.metaspectra["bad_data"][0])[0]
            ppxf_fitter = ppxf.ppxf(matched_library.spectrumset.spectra.T, # templates in cols
                               target_spec.spectra[0],
                               target_spec.metaspectra["noise"][0],
                               velscale, guess,
                               goodpixels=good_pixels_indicies,
                               bias=bias, moments=num_moments,
                               degree=add_deg, mdegree=mul_deg,
                               vsyst=velocity_offset, plot=False, quiet=True)
            results[bin_iter, 0] = bin_num
            results[bin_iter, 1] = ppxf_fitter.chi2
            results[bin_iter, 2:] = ppxf_fitter.sol
        return np.asarray(results)

