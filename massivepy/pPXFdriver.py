"""
This is a wrapper for ppxf, facilitating kinematic fitting.
"""


import warnings
import functools
import copy

import numpy as np
import scipy.signal as signal
import scipy.integrate as integ
import ppxf
import matplotlib.pyplot as plt

import utilities as utl
import massivepy.constants as const
import massivepy.spectrum as spec
import massivepy.templates as temp
import massivepy.gausshermite as gh


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
    PPXF_REQUIRED_INPUTS = {"add_deg":     ["degree", int],
                            "mul_deg":     ["mdegree", int],
                            "num_moments": ["moments", int],
                            "bias":        ["bias", float]}
        # These are input parameter names and types needed by pPXF's caller

    def __init__(self, specset=None, templib=None, fit_range=None,
                 initial_gh=None, num_trials=None, **ppxf_kwargs):
        """
        Args:
        """
        # validate input types
        self.specset = copy.deepcopy(spectra)
        self.templib = copy.deepcopy(templates)
        self.nominal_fit_range = np.asarray(fit_range, dtype=float)
        if self.nominal_fit_range.shape != (2,):
            raise ValueError("invalid fit range shape {}"
                             "".format(self.nominal_fit_range.shape))
        self.num_trials = int(num_trials)
        if not (self.num_trials => 0):
            raise ValueError("invalid number of Monte Carlo trials: "
                             "{}, must be a non-negative integer"
                             "".format(self.num_trials))
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
            raise ValueError("invalid starting gh parameters shape {}, "
                             "must match number of moments to fit {}"
                             "".format(self.initial_gh.shape,
                                       self.ppxf_kwargs["moments"]))
        # output containers
        self.main_input = {}
        self.main_rawoutput = {}
        self.main_procoutput = {}
        self.mc_input = {}
        self.mc_rawoutput = {}
        self.mc_procoutput = {}
        # prep spectra
        print ("preparing {} spectra for fitting..."
               "".format(self.specset.num_spectra))
        try:
            self.specset.log_resample()
        except ValueError, msg:
            print "skipping spectra log_resample ({})".format(msg)
        self.specset = self.specset.crop(self.nominal_fit_range)
        self.exact_fit_range = utl.min_max(self.specset.waves)
        self.logscale = self.specset.get_logscale()
        self.velscale = self.logscale*const.c_kms
        self.specset = self.specset.get_normalized(
            norm_func=spec.SpectrumSet.compute_spectrum_median,
            norm_value=1.0)  # normalize only for sensible numerical values

    def prepare_library(self, target_spec):
        """
        """
        spec_waves = target_spec.waves
        spec_ir = target_spec.metaspectra["ir"][0]
        spec_ir_inerp_func = utl.interp1d_constextrap(spec_waves, spec_ir)
            # use to get ir(template_waves); this will require some
            # extrapolation beyond the spectra wavelength range, which
            # the above function will do as a constant
        template_waves = self.templib.spectrumset.waves
        single_ir_to_match = spec_ir_inerp_func(template_waves)
        num_temps = self.templib.spectrumset.num_spectra
        ir_to_match = np.asarray([single_ir_to_match,]*num_temps)
        matched_library = self.templib.match_resolution(ir_to_match)
        matched_library.spectrumset.log_resample(self.logscale)
        matched_library.spectrumset = (
            matched_library.spectrumset.get_normalized(
                norm_func=spec.SpectrumSet.compute_spectrum_median,
                norm_value=1.0))
        return matched_library

    def get_pPXF_inputdict(ppxf_fitter):
        """
        """
        raw_inputs = {}
        raw_inputs["pixels_used"] = ppxf_fitter.goodpixels
        raw_inputs["bias"] = ppxf_fitter.bias
        raw_inputs["lam"] = ppxf_fitter.lam
        raw_inputs["mul_deg"] = ppxf_fitter.mdegree
        raw_inputs["reg_dim"] = ppxf_fitter.reg_dim
        raw_inputs["clean"] = ppxf_fitter.clean
        raw_inputs["num_moments"] = ppxf_fitter.moments
        raw_inputs["regul"] = ppxf_fitter.regul
        raw_inputs["sky_template"] = ppxf_fitter.sky
        raw_inputs["add_deg"] = ppxf_fitter.degree
        raw_inputs["noise"] = ppxf_fitter.noise
        raw_inputs["templates"] = ppxf_fitter.star.T
        raw_inputs["oversample"] = ppxf_fitter.oversample
        raw_inputs["vsyst"] = ppxf_fitter.vsyst*self.velscale
        raw_inputs["spectrum"] = ppxf_fitter.galaxy
        raw_inputs["kin_components"] = ppxf_fitter.component
        raw_inputs["sampling_factor"] = ppxf_fitter.factor
        return raw_inputs

    def get_pPXF_rawoutputdict(ppxf_fitter):
        """
        """
        raw_outputs = {}
        raw_outputs["best_model"] = ppxf_fitter.bestfit
        raw_outputs["reddening"] = ppxf_fitter.reddening
        raw_outputs["chisq_dof"] = ppxf_fitter.chi2
        raw_outputs["num_kin_components"] = ppxf_fitter.ncomp
        raw_outputs["gh_params"] = ppxf_fitter.sol
        raw_outputs["unscaled_lsq_errors"] = ppxf_fitter.error
            # error estimate from least-squares cov matrix
        raw_outputs["add_weights"] = ppxf_fitter.polyweights
        raw_outputs["template_weights"] = ppxf_fitter.weights
        try:
            raw_outputs["mul_weights"] = (
                np.concatenate(([1], ppxf_fitter.mpolyweights)))
                # constant term is always 1, but not returned by pPXF
        except AttributeError:
            raw_outputs["mul_weights"] = np.asarray([1])
                # if no mult poly specified in the fit, then pPXF still uses
                # a constant term of 1, but returns no output mult polynomial
        return raw_outputs

    def scale_lsqvariance(self, ppxf_fitter):
        """
        Scale the error estimate derived from the least-squares
        algorithm's covariance matrix by sqrt(chisq). This provides a
        more realistic estimate when the fit is poor, though in either
        case it is superseded by Monte Carlo errors.
        """
        return ppxf_fitter.error*np.sqrt(ppxf_fitter.chi2)

    def evalute_continuum_polys(self, ppxf_fitter):
        """
        Evaluate both the multiplicative and additive continuum
        polynomials using the input spectrum's wavelength sampling
        """
        poly_args = np.linspace(-1, 1, ppxf_fitter.galaxy.shape[0])
            # pPXF evaluates polynomials by mapping the fit log-lambda
            # interval linearly onto the Legendre interval [-1, 1]
        legendre_series = np.polynomial.legendre.legval
        mul_continuum = legendre_series(poly_args,
                                          raw_outputs["mul_weights"])
        add_continuum = legendre_series(poly_args, ppxf_fitter.polyweights)
        proc_outputs["mul_continuum"] = mul_continuum
        proc_outputs["add_continuum"] = add_continuum
        return

    def compute_convolved_templates():
        bf_v, bf_sigma = raw_outputs["gh_params"][:2]
        rough_edge = (np.absolute(raw_inputs["vsyst"]) +
                      np.absolute(bf_v) + sampling_scale*bf_sigma)
        num_steps = np.ceil(rough_edge/self.velscale)
        exact_edge = self.velscale*num_steps
            # sample losvd in [-exact_edge, exact_edges], steps of logscale*c
        kernel_size = 2*num_steps + 1
        velocity_samples = np.linspace(-exact_edge, exact_edge, kernel_size)
        shifted_gh_params = np.concatenate(([bf_v + raw_inputs["vsyst"]],
                                             raw_outputs["gh_params"][1:]))
        shifted_losvd = gh.unnormalized_gausshermite_pdf(velocity_samples,
                                                         shifted_gh_params)
        smoothed_temps = np.zeros(raw_inputs["templates"].shape)
        model_temps = np.zeros((raw_inputs["templates"].shape[0],
                                raw_inputs["spectrum"].shape[0]))
        model_temps_fluxes = np.zeros(raw_inputs["templates"].shape[0])
        temp_waves = temp_lib.spectrumset.waves
        for temp_iter, temp in enumerate(raw_inputs["templates"]):
            convolved = signal.fftconvolve(temp, shifted_losvd, mode='same')*self.velscale
            smoothed_temps[temp_iter, :] = convolved
            model_temps[temp_iter, :] = convolved[:mul_continuum.shape[0]]*mul_continuum
            flux = integ.simps(model_temps[temp_iter, :], self.specset.waves)
            model_temps_fluxes[temp_iter] = flux
        proc_outputs["smoothed_temps"] = smoothed_temps
        proc_outputs["model_temps"] = model_temps
        proc_outputs["model_temps_fluxes"] = model_temps_fluxes
        fit_width = self.exact_fit_range[1] - self.exact_fit_range[1]
        total_flux = integ.simps(raw_outputs["best_model"],
                                 self.specset.waves)
        proc_outputs["total_flux"] = total_flux
        flux_add_weights = raw_outputs["add_weights"]*fit_width/total_flux
        proc_outputs["flux_add_weights"] = flux_add_weights
        flux_template_weights = (
            raw_outputs["template_weights"]*model_temps_fluxes/total_flux)
        proc_outputs["flux_template_weights"] = flux_template_weights
        model_reconstruct = (add_continuum +
            np.sum(model_temps.T*raw_outputs["template_weights"], axis=1))
        full_template = np.sum(model_temps.T*raw_outputs["template_weights"], axis=1)
        model_delta = model_reconstruct - raw_outputs['best_model']
        model_frac_delta = model_delta/raw_outputs['best_model']
        matches = np.absolute(model_frac_delta).max() < const.float_tol
        if not matches:
            fig, ax = plt.subplots()
            ax.plot(self.specset.waves, raw_outputs['best_model'],
                    linestyle='-', marker='', color='k', alpha=0.6,
                    label="pPXF-returned best-fit model")
            ax.plot(self.specset.waves, model_reconstruct,
                    linestyle='-', marker='', color='r', alpha=0.6,
                    label="reconstructed best-fit model")
            plt.show()
            raise RuntimeError("reconstructed spectrum model does not match")
        return raw_inputs, raw_outputs, proc_outputs


    def process_pPXF_results(self, ppxf_fitter, temp_lib, sampling_scale):
        """
        """
        proc_outputs = {}
        error_scale = np.sqrt(ppxf_fitter.chi2)
        proc_outputs["scaled_lsq_errors"] = ppxf_fitter.error*error_scale
            # error estimate from least-squares cov matrix, scaled by
            # root-chisq to account for residuals in poorly-fit data
        poly_args = np.linspace(-1, 1, raw_inputs["spectrum"].shape[0])
            # pPXF evaluates polynomials by mapping the fit log-lambda
            # interval linearly onto the Legendre interval [-1, 1]
        mul_continuum = (
            np.polynomial.legendre.legval(poly_args,
                                          raw_outputs["mul_weights"]))
        add_continuum = (
            np.polynomial.legendre.legval(poly_args,
                                          raw_outputs["add_weights"]))
        proc_outputs["mul_continuum"] = mul_continuum
        proc_outputs["add_continuum"] = add_continuum
        bf_v, bf_sigma = raw_outputs["gh_params"][:2]
        rough_edge = (np.absolute(raw_inputs["vsyst"]) +
                      np.absolute(bf_v) + sampling_scale*bf_sigma)
        num_steps = np.ceil(rough_edge/self.velscale)
        exact_edge = self.velscale*num_steps
            # sample losvd in [-exact_edge, exact_edges], steps of logscale*c
        kernel_size = 2*num_steps + 1
        velocity_samples = np.linspace(-exact_edge, exact_edge, kernel_size)
        shifted_gh_params = np.concatenate(([bf_v + raw_inputs["vsyst"]],
                                             raw_outputs["gh_params"][1:]))
        shifted_losvd = gh.unnormalized_gausshermite_pdf(velocity_samples,
                                                         shifted_gh_params)
        smoothed_temps = np.zeros(raw_inputs["templates"].shape)
        model_temps = np.zeros((raw_inputs["templates"].shape[0],
                                raw_inputs["spectrum"].shape[0]))
        model_temps_fluxes = np.zeros(raw_inputs["templates"].shape[0])
        temp_waves = temp_lib.spectrumset.waves
        for temp_iter, temp in enumerate(raw_inputs["templates"]):
            convolved = signal.fftconvolve(temp, shifted_losvd, mode='same')*self.velscale
            smoothed_temps[temp_iter, :] = convolved
            model_temps[temp_iter, :] = convolved[:mul_continuum.shape[0]]*mul_continuum
            flux = integ.simps(model_temps[temp_iter, :], self.specset.waves)
            model_temps_fluxes[temp_iter] = flux
        proc_outputs["smoothed_temps"] = smoothed_temps
        proc_outputs["model_temps"] = model_temps
        proc_outputs["model_temps_fluxes"] = model_temps_fluxes
        fit_width = self.exact_fit_range[1] - self.exact_fit_range[1]
        total_flux = integ.simps(raw_outputs["best_model"],
                                 self.specset.waves)
        proc_outputs["total_flux"] = total_flux
        flux_add_weights = raw_outputs["add_weights"]*fit_width/total_flux
        proc_outputs["flux_add_weights"] = flux_add_weights
        flux_template_weights = (
            raw_outputs["template_weights"]*model_temps_fluxes/total_flux)
        proc_outputs["flux_template_weights"] = flux_template_weights
        model_reconstruct = (add_continuum +
            np.sum(model_temps.T*raw_outputs["template_weights"], axis=1))
        full_template = np.sum(model_temps.T*raw_outputs["template_weights"], axis=1)
        model_delta = model_reconstruct - raw_outputs['best_model']
        model_frac_delta = model_delta/raw_outputs['best_model']
        matches = np.absolute(model_frac_delta).max() < const.float_tol
        if not matches:
            fig, ax = plt.subplots()
            ax.plot(self.specset.waves, raw_outputs['best_model'],
                    linestyle='-', marker='', color='k', alpha=0.6,
                    label="pPXF-returned best-fit model")
            ax.plot(self.specset.waves, model_reconstruct,
                    linestyle='-', marker='', color='r', alpha=0.6,
                    label="reconstructed best-fit model")
            plt.show()
            raise RuntimeError("reconstructed spectrum model does not match")
        return raw_inputs, raw_outputs, proc_outputs

    def run_fit(self):
        """
        """
        if self.main_rawoutput:
            raise RuntimeWarning("A pPXF fit to this set of spectra has "
                                 "been computed - overwriting...")
        for spec_iter, target_id in enumerate(self.specset.ids):
            target_spec = self.specset.get_subset([target_id])
            matched_library = self.prepare_library(target_spec)
            template_range = utl.min_max(matched_library.spectrumset.waves)
            log_temp_start = np.log(template_range[0])
            log_spec_start = np.log(self.exact_fit_range[0])
            velocity_offset = (log_temp_start - log_spec_start)*const.c_kms
                # pPXF assumes that the data and templates are sampled with
                # identical log-steps and starting wavelengths. If this is
                # not the case, the output best-fit velocity parameter will
                # be increased by an amount corresponding to the difference
                # between the data and templates initial sampling wavelengths.
                # pPXF will automatically subtract this extra shift if passed
                # the velocity-difference between the initial wavelengths.
            good_pix_mask = ~target_spec.metaspectra["bad_data"][0]
            good_pix_indices = np.where(good_pix_mask)[0]
                # np.where outputs tuple for some reason: (w,), with w being
                # an array containing integer indices where the input is True
            library_spectra_cols = matched_library.spectrumset.spectra.T
                # pPXF requires library spectra in columns of input array
            raw_fitter = ppxf.ppxf(library_spectra_cols,
                                   target_spec.spectra[0],
                                   target_spec.metaspectra["noise"][0],
                                   self.velscale, self.initial_gh,
                                   goodpixels=good_pix_indices,
                                   vsyst=velocity_offset, plot=False,
                                   quiet=True, **self.ppxf_kwargs)

            # save some results
            if self.num_trials > 1:
                # generate Monte Carlo trial spectra
                base = self.main_rawoutput("best_model")
                chsq_dof = self.main_rawoutput("chisq_dof")
                if chsq_dof >= 1:
                    raise Warning("chi^2/dof = {} >= 1\n"
                                  "inflating noise for Monte Carlo trials "
                                  "such that chi^2/dof = 1".format(chsq_dof))
                else:
                    raise Warning("chi^2/dof = {} < 1\n"
                                  "deflating noise for Monte Carlo trials "
                                  "such that chi^2/dof = 1".format(chsq_dof))
                noise_scale = self.main_input["noise"]*np.sqrt(chisq_dof)
                    # this sets the noise level for the Monte Carlo trials to
                    # be roughly the size of the actual fit residuals - i.e.,
                    # we assume the model is valid and that a high/low chisq
                    # per dof is due to under/overestimated errors
                noise_draw = np.random.randn(self.num_trials, *base.shape)
                    # uniform, uncorrelated Gaussian pixel noise
                trial_spectra = base + noise_draw*noise_scale
                base_gh_params = self.main_rawoutput["gh_params"]
                for trial_spectrum in trial_spectra:
                    trial_fitter = ppxf.ppxf(library_spectra_cols,
                                             trial_spectrum, noise_draw,
                                             self.velscale, base_gh_params,
                                             goodpixels=good_pix_indices,
                                             vsyst=velocity_offset,
                                             plot=False, quiet=True,
                                             **self.ppxf_kwargs)


                bin_output["spectrum"][trial, ...] = simulated_galaxy
        return raw_fitter

