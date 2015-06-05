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
import massivepy.templates as temp


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
    PPXF_REQUIRED_INPUTS = {"add_deg": ["degree", int],
                            "mul_deg": ["mdegree", int],
                            "num_moments": ["moments", int],
                            "bias": ["bias", float]}
        # These are input parameter names and types needed by pPXF's caller

    def __init__(self, spectra=None, templates=None,
                 fit_range=None, initial_gh=None, **ppxf_kwargs):
        """
        Args:
        """
        self.spectra = copy.deepcopy(spectra)
        self.templates = copy.deepcopy(templates)
        self.nominal_fit_range = np.asarray(fit_range, dtype=float)
        if self.nominal_fit_range.shape != (2,):
            raise ValueError("Invalid fit range shape {}"
                             "".format(self.nominal_fit_range.shape))
        for setting in self.PPXF_REQUIRED_INPUTS:
            if setting not in ppxf_kwargs:
                raise ValueError("{} must be specified to fit spectra"
                                 "".format(setting))
        self.ppxf_kwargs = {}
        for setting, value in ppxf_kwargs.iteritems():
            ppxf_name, conversion = self.PPXF_REQUIRED_INPUTS[setting]
            self.ppxf_kwargs[ppxf_name] = conversion(value)
        self.initial_gh = np.asarray(initial_gh, dtype=float)
        if self.initial_gh.shape != (self.ppxf_kwargs["moments"],):
            raise ValueError("Invalid starting gh parameters shape {}, "
                             "must match number of moments to fit {}"
                             "".format(self.initial_gh.shape,
                                       self.ppxf_kwargs["moments"]))
        self.results = None
        # prepare copied spectra
        try:
            self.spectra.log_resample()
        except ValueError, msg:
            print "skipping spectra log_resample ({})".format(msg)
        self.to_fit = utl.in_linear_interval(self.spectra.waves,
                                             self.nominal_fit_range)
        self.spectra = self.spectra.crop(self.nominal_fit_range)
        self.exact_fit_range = utl.min_max(self.spectra.waves)
        self.spectra = self.spectra.get_normalized(
            norm_func=spec.SpectrumSet.compute_spectrum_median,
            norm_value=1.0)

    def prepare_library(self, target_spec):
        """
        """
        spec_waves = target_spec.waves
        spec_ir = target_spec.metaspectra["ir"][0]
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
        logscale = target_spec.get_logscale()
        matched_library.spectrumset.log_resample(logscale)
        # norm
        matched_library.spectrumset = (
            matched_library.spectrumset.get_normalized(
                norm_func=spec.SpectrumSet.compute_spectrum_median,
                norm_value=1.0))
        return matched_library

    def normalize_output(self, ppxf_fitter):
        """
        """
        results, called_with = {}, {}
        called_with["pixels_used"] = ppxf_fitter.goodpixels
        called_with["bias"] = ppxf_fitter.bias
        called_with["lam"] = ppxf_fitter.lam
        called_with["mul_deg"] = ppxf_fitter.mdegree
        called_with["reg_dim"] = ppxf_fitter.reg_dim
        called_with["clean"] = ppxf_fitter.clean
        called_with["num_moments"] = ppxf_fitter.moments
        called_with["regul"] = ppxf_fitter.regul
        called_with["sky_template"] = ppxf_fitter.sky
        called_with["add_deg"] = ppxf_fitter.degree
        called_with["noise"] = ppxf_fitter.noise
        called_with["templates"] = ppxf_fitter.star.T
        called_with["oversample"] = ppxf_fitter.oversample
        called_with["vsyst"] = ppxf_fitter.vsyst
        called_with["spectrum"] = ppxf_fitter.galaxy
        results["best_model"] = ppxf_fitter.bestfit
        results["reddening"] = ppxf_fitter.reddening
        results["chisq_dof"] = ppxf_fitter.chi2
        results["kin_component"] = ppxf_fitter.component
        results["mul_weights"] = np.concatenate(
            ([1], ppxf_fitter.mpolyweights))
            # constant term is always 1, but not returned by pPXF
        results["num_kin_components"] = ppxf_fitter.ncomp
        results["gh_parameters"] = ppxf_fitter.sol
        results["sampling_factor"] = ppxf_fitter.factor
        results["add_weights"] = ppxf_fitter.polyweights
        results["template_weights"] = ppxf_fitter.weights
        results["lsq_error"] = ppxf_fitter.error*np.sqrt(ppxf_fitter.chi2)
            # scale leastsq error estimate for poor fits
        return results, called_with

    def run_fit(self):
        """
        """
        output = []
        for spec_iter, target_id in enumerate(self.spectra.ids):
            target_spec = self.spectra.get_subset([target_id])
            matched_library = self.prepare_library(target_spec)
            template_range = utl.min_max(matched_library.spectrumset.waves)
            log_temp_start = np.log(template_range[0])
            log_spec_start = np.log(self.exact_fit_range[0])
            velocity_offset = (log_temp_start - log_spec_start)*const.c_kms
            logscale = target_spec.get_logscale()
            velscale = logscale*const.c_kms
            good_pix_mask = ~target_spec.metaspectra["bad_data"][0]
            good_pix_indices = np.where(good_pix_mask)[0]
                # np.where outputs tuple for some reason: (w,), with w being
                # an array containing integer indices where the input is True
            library_spectra_cols = matched_library.spectrumset.spectra.T
                # pPXF requires library spectra in columns of input array
            raw_fitter = ppxf.ppxf(library_spectra_cols,
                                   target_spec.spectra[0],
                                   target_spec.metaspectra["noise"][0],
                                   velscale, self.initial_gh,
                                   goodpixels=good_pix_indices,
                                   vsyst=velocity_offset, plot=False,
                                   quiet=True, **self.ppxf_kwargs)
            self.results = self.normalize_output(raw_fitter)
        return raw_fitter

