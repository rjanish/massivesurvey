"""
This is a wrapper for ppxf, facilitating kinematic fitting.
"""

import os
import warnings
import functools
import copy

import numpy as np
import scipy.signal as signal
import scipy.integrate as integ
import ppxf
import matplotlib.pyplot as plt
import astropy.io.fits as fits

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
    VEL_FACTOR = const.ppxf_losvd_sampling_factor

    def __init__(self, specset=None, templib=None, fit_range=None,
                 initial_gh=None, num_trials=None, sourcefile=None,
                 sourcedate=None, **ppxf_kwargs):
        """
        This validates the passed pPXF settings, and prepares the
        input spectra for fitting.

        Copies are made of the input spectral dataset and template
        library, to ensure that any manipulations required before
        fitting to do propagate back to original data.

        Args:
        specset - massivepy.SpectrumSet object
            The spectral data to fit
        templib - massivepy.TemplateLibrary object
            The library of stellar template to use in the fitting
        fit_range - (2,) arraylike
            The wavelength interval over which to fit the spectra
        initial_gh - 1d arraylike
            The Gauss-Hermite parameters with which to start the fit
        num_trials - int
            The number of Monte Carlo trials to run in order to
            determine the errors in the output fit parameters
        sourcefile - str
            The file name of the fits file containing the spectra to
            be fit, for metadata tracking purposes
        sourcedate - str
            The last modified date of the sourcefile
        add_deg - int
            The degree of the additive Legendre polynomial series
            used to model the continuum
        mul_deg - int
            The degree minus 1 of the multiplicative Legendre
            polynomial series used to model the continuum. The constant
            term is always a 1, and this input specifies the number
            of additional terms to include.
        num_moments - int
            The number of Gauss-Hermite moments to include in the fit
        bias - float, positive
            The Gaussian biasing parameters - see pPXF paper
        """
        # validate input types
        self.specset = copy.deepcopy(specset)
        self.templib = copy.deepcopy(templib)
        self.nominal_fit_range = np.asarray(fit_range, dtype=float)
        if self.nominal_fit_range.shape != (2,):
            raise ValueError("invalid fit range shape {}"
                             "".format(self.nominal_fit_range.shape))
        self.num_trials = int(num_trials)
        if not (self.num_trials >= 0):
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
        self.sourcefile = sourcefile
        self.sourcedate = sourcedate
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
        # prep fit trackers
        self.fit_complete = False

    def init_output_containers(self, debug_mode=False):
        """
        Prepare containers for all pPXF outputs. This is where the structure
        of the output fits files is determined.

        Changing debug_mode to True results in keeping the full list of hdu
        names for later saving to file by write_outputs; the default
        debug_mode = False sets up a shortened list of hdu names to save to
        file. This shortened list leaves out all storage intensive items
        (which in practice means anything shaped like a spectrum) unless
        they are routinely used in the normal plotting/analysis.

        Note that regardless of debug_mode setting, the full list of outputs
        is initialized and saved to the output containers, as debug_mode
        only impacts what will be saved to file.
        """
        # shorthand for all of the relevant dimensions
        n_mom = self.ppxf_kwargs['moments']
        n_spec = self.specset.num_spectra
        n_pix = self.specset.num_samples
        n_mc = self.num_trials
        n_addw = self.ppxf_kwargs['degree'] + 1  # deg 0 --> 1 const
        n_mulw = self.ppxf_kwargs['mdegree'] + 1
        n_temp = self.num_temps
        n_temppix = self.num_temp_samples
        # convolved template libraries, separate from output to save
        temp_lib_shape = [n_spec, n_temp, n_temppix]
        self.matched_templates = {"spectra":  np.zeros(temp_lib_shape),
                                  "ir":  np.zeros(temp_lib_shape),
                                  "waves":  np.zeros(n_temppix)}
        # output that will be saved to fits, grouped by data shape
        self.output_hdunames = ["moments","templates","spectrum","scalars",
                                "waves","add_weights","mul_weights",
                                "smoothed_templates","model_templates"]
        self.mcoutput_hdunames = ["moments","templates","spectrum","scalars",
                                  "waves","add_weights","mul_weights",
                                  "smoothed_templates","model_templates",
                                  "noiselevels"]
        self.output_names = {"scalars": ["binid","chisq_dof"],
                             "moments": ["gh_params","unscaled_lsq_errors",
                                            "scaled_lsq_errors"],
                             # unscaled_lsq is estimate from least-square fit's
                             #  covariance matrix, scaled_lsq scales by chisq.
                             "templates": ["template_ids","template_weights",
                                          "model_temps_fluxes",
                                          "template_fluxweights"],
                             "spectrum": ["best_model","mul_poly","add_poly"],
                             "add_weights": ["add_weights","add_fluxweights"],
                             # add_fluxweights are weighted normalied to equal
                             #  fractional flux level
                             "mul_weights": ["mul_weights"],
                             "smoothed_templates": ["smoothed_temps"],
                             # smoothed by losvd
                             "model_templates": ["model_temps"],
                             # also scaled by mulpoly
                             "waves": ["waves"]}
        # self.mcoutput_names is nearly the same, but we save 2 extra items
        self.mcoutput_names = copy.deepcopy(self.output_names)
        self.mcoutput_names["spectrum"].append("spectrum")
        self.mcoutput_names["noiselevels"] = ["noiselevels"]
        output_shapes = {"scalars": (n_spec),
                         "moments": (n_spec,n_mom),
                         "templates": (n_spec,n_temp),
                         "spectrum": (n_spec,n_pix),
                         "add_weights": (n_spec,n_addw),
                         "mul_weights": (n_spec,n_mulw),
                         "smoothed_templates": (n_spec,n_temp,n_temppix),
                         "model_templates": (n_spec,n_temp,n_pix),
                         "waves": (n_pix)}
        mc_shapes = {"scalars": (n_spec,n_mc),
                     "moments": (n_spec,n_mc,n_mom),
                     "templates": (n_spec,n_mc,n_temp),
                     "spectrum": (n_spec,n_mc,n_pix),
                     "add_weights": (n_spec,n_mc,n_addw),
                     "mul_weights": (n_spec,n_mc,n_mulw),
                     "smoothed_templates": (n_spec,n_mc,n_temp,n_temppix),
                     "model_templates": (n_spec,n_mc,n_temp,n_pix),
                     "waves": (n_pix),
                     "noiselevels": (n_spec,n_pix)}
        # now create the output containers
        self.bestfit_output = {}
        for hduname in self.output_hdunames:
            shape = output_shapes[hduname]
            for outputname in self.output_names[hduname]:
                self.bestfit_output[outputname] = np.zeros(shape)
        self.mc_output = {}
        for hduname in self.mcoutput_hdunames:
            shape = mc_shapes[hduname]
            for outputname in self.mcoutput_names[hduname]:
                self.mc_output[outputname] = np.zeros(shape)
        if not debug_mode:
            self.output_hdunames.remove("smoothed_templates")
            self.output_hdunames.remove("model_templates")
            self.output_names["spectrum"].remove("add_poly")
            self.output_names["spectrum"].remove("mul_poly")
            self.mcoutput_hdunames.remove("smoothed_templates")
            self.mcoutput_hdunames.remove("model_templates")
            self.mcoutput_hdunames.remove("noiselevels")
            self.mcoutput_hdunames.remove("spectrum")
            self.mcoutput_hdunames.remove("waves")
        return

    def save_matched_templates(self, matched_lib, index):
        """
        Save copy of the ir-matched template spectra, ir, and wavelengths.
        All other TemplateLibrary info is the same for each bin. 
        """
        per_bin = {"spectra": matched_lib.spectrumset.spectra,
                   "ir": matched_lib.spectrumset.metaspectra["ir"]}
        utl.fill_dict(self.matched_templates, per_bin, index)
        first_update = (index == 0)
        if first_update:
            self.matched_templates['waves'] = matched_lib.spectrumset.waves
        else:
            resid = np.absolute(self.matched_templates['waves'] -
                                matched_lib.spectrumset.waves)
            waves_match = np.all(resid < const.float_tol)
            if not waves_match:
                raise RuntimeError("IR-matched template library sampling "
                                   "does not agree across spectra to fit!")
            self.matched_templates['waves'] = matched_lib.spectrumset.waves

    def prepare_library(self, target_spec):
        """
        Prepare a template library based on the library in pPXFdriver's
        templib attribute to be usable to fit the passed SpectrumSet.
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
        matched_library.spectrumset = (
            matched_library.spectrumset.crop(self.valid_temp_range))
        matched_library.spectrumset.log_resample(self.logscale)
        matched_library.spectrumset = (
            matched_library.spectrumset.get_normalized(
                norm_func=spec.SpectrumSet.compute_spectrum_median,
                norm_value=1.0))
            # normalize by median only for sensible numerical values, as a
            # proper flux-normalization in the fitting region can only be
            # done after the best-fit overall velocity shift is determined
        return matched_library

    def prepend_const_mweight(self, ppxf_fitter):
        """
        The multiplicative polynomial in pPXF has a constant term of
        fixed coefficient 1, but pPXF only returns the variable
        coefficients (for the linear and higher-order terms). This
        restores the 1, so that additive and multiplicative polynomial
        weights are reported identically.
        """
        try:
            mul_weights = np.concatenate(([1], ppxf_fitter.mpolyweights))
        except AttributeError:
            mul_weights = np.asarray([1])
                # 'mpolyweights' is not defined in pPXF if only a constant
        return mul_weights

    def scale_lsq_variance(self, ppxf_fitter):
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
        add_continuum = legendre_series(poly_args, ppxf_fitter.polyweights)
        proc_mul_weights = self.prepend_const_mweight(ppxf_fitter)
        mul_continuum = legendre_series(poly_args, proc_mul_weights)
        return add_continuum, mul_continuum

    def convolve_templates(self, ppxf_fitter, losvd_sampling_factor):
        """
        Convolve the templates by the best-fitting losvd. The velocity
        width over which the losvd is sampled its central velocity +-
        losvd_sampling_factor*sigma. The losvds used here are *not*
        normalized unless exactly Gaussian (this is in accord with
        pPXF's convention), so the total fluxes will change here.

        The returned smoothed templates are sampled with the same
        starting wavelength and log-step as the data, but possibly
        extending to some much greater final wavelength. This is
        ensured by accounting for the difference in template and data
        initial sampling values in the fit using pPXF's vsyst.
        """
        v, sigma = ppxf_fitter.sol[:2]
        vsyst = ppxf_fitter.vsyst*self.velscale
            # the template to data initial sampling shift - pPXF saves
            # the value in units of pixels, restore velocity units here
        # get velocity sampling
        rough_edge = (np.absolute(vsyst) +
                      np.absolute(v) +
                      losvd_sampling_factor*sigma)
        num_steps = np.ceil(rough_edge/self.velscale)
            # number of steps from zero to outer edge of sampling
        exact_edge = self.velscale*num_steps
            # allow sampling of [-exact_edge, exact_edges] via velscale steps
        kernel_size = 2*num_steps + 1  # center kernel on zero
        velocity_samples = np.linspace(-exact_edge, exact_edge, kernel_size)
        # compute losvd
        shifted_gh_params = np.concatenate(([v + vsyst],
                                            ppxf_fitter.sol[1:]))
        shifted_losvd = gh.unnormalized_gausshermite_pdf(velocity_samples,
                                                         shifted_gh_params)
            # this includes the difference in initial sampling wavelengths
            # between the templates and data - using this losvd is accurate
            # if one assumed that templates and data are identically sampled
        # do convolutions
        template_spectra = ppxf_fitter.star.T # ppxf stores spectra in cols
        smoothed_temps = np.zeros(template_spectra.shape)
        for temp_iter, temp in enumerate(template_spectra):
            convolved = self.velscale*signal.fftconvolve(temp, shifted_losvd,
                                                         mode='same')
                # factor of velscale makes discrete convolution ~ trapezoid
                # approximation to continuous convolution integral, see notes
            smoothed_temps[temp_iter, :] = convolved
        return smoothed_temps

    def compute_model_templates(self, fitter, smoothed_templates, mul_poly):
        """
        The computes the 'model templates', meaning the template terms
        as the appear in the final model summation - the templates
        convolved by the losvd and scaled by the multiplicative
        polynomial, and then sampled identical to the data spectrum.
        Also returns the integrated (over wavelength) fluxes of the
        model templates.
        """
        num_templates = fitter.star.shape[1] # ppxf stores templates in cols
        num_data_samples = fitter.galaxy.shape[0]
        model_temps = np.zeros((num_templates, num_data_samples))
        fluxes = np.zeros(num_templates)
        for temp_iter, smoothed in enumerate(smoothed_templates):
            model_temps[temp_iter, :] = smoothed[:num_data_samples]*mul_poly
                # the smoothed templates are sampled with the same starting
                # wavelength and log-step as the data, but possibly extending
                # to some much greater final wavelength
            fluxes[temp_iter] = integ.simps(model_temps[temp_iter, :],
                                            self.specset.waves)
        return model_temps, fluxes

    def fluxnormalize_weights(self, fitter, model_tmps_fluxes):
        """
        Normalize the additive polynomial and template weights such
        that they equal the fractional flux contribution of their
        respective terms to the total model flux. This is done using
        the total fluxes integrated over wavelength.

        Only the constant term in the additive polynomial technically
        carries a total flux, so it is normalized as above and then
        the remaining terms are normalized to have identical units.
        This means that $a_0 + \sum_i w_i = 1$, for returned a and w.
        """
        fit_wave_width = self.exact_fit_range[1] - self.exact_fit_range[0]
        total_flux = integ.simps(fitter.bestfit, self.specset.waves)
        flux_add_weights = fitter.polyweights*fit_wave_width/total_flux
        flux_template_weights = fitter.weights*model_tmps_fluxes/total_flux
        return flux_add_weights, flux_template_weights

    def get_raw_pPXF_results(self, ppxf_fitter):
        """
        Compile useful pPXF direct outputs.
        """
        raw_outputs = {}
        # raw ppxf outputs
        raw_outputs["best_model"] = ppxf_fitter.bestfit
            # best-fit model spectrum, sampled identically to input data
        raw_outputs["gh_params"] = ppxf_fitter.sol
        raw_outputs["chisq_dof"] = ppxf_fitter.chi2 # is per dof, see ppxf.py
        raw_outputs["add_weights"] = ppxf_fitter.polyweights
        raw_outputs["template_weights"] = ppxf_fitter.weights
        raw_outputs["mul_weights"] = self.prepend_const_mweight(ppxf_fitter)
            # multiplicative weights need constant term
        raw_outputs["unscaled_lsq_errors"] = ppxf_fitter.error
            # error estimate from least-square algorithm's covariance matrix
        return raw_outputs

    def process_pPXF_results(self, fitter, losvd_sampling_factor):
        """
        Compute useful outputs not directly returned by pPXF.
        """
        proc_outputs = {}
        proc_outputs["template_ids"] = self.templib.spectrumset.ids
        proc_outputs["scaled_lsq_errors"] = self.scale_lsq_variance(fitter)
            # least-square error estimate scaled by fit chi-squared
        add_poly, mul_poly = self.evalute_continuum_polys(fitter)
        proc_outputs["mul_poly"] = mul_poly
        proc_outputs["add_poly"] = add_poly
        smoothed_tmps = self.convolve_templates(fitter, losvd_sampling_factor)
        proc_outputs["smoothed_temps"] = smoothed_tmps
            # templates convolved by best-fit losvd
        [model_tmps, model_fluxes] = (
            self.compute_model_templates(fitter, smoothed_tmps, mul_poly))
        proc_outputs["model_temps"] = model_tmps
            # losvd-smoothed templates also scaled by multiplicative poly
        proc_outputs["model_temps_fluxes"] = model_fluxes
        [flux_add_weights,
         flux_tmp_weights] = self.fluxnormalize_weights(fitter, model_fluxes)
        proc_outputs["add_fluxweights"] = flux_add_weights
        proc_outputs["template_fluxweights"] = flux_tmp_weights
        # test best-fit model reconstruction
        reconstructed = add_poly + (model_tmps.T*fitter.weights).sum(axis=1)
        delta = (reconstructed - fitter.bestfit)/fitter.bestfit
        #matches = np.absolute(delta).max() < const.relaxed_tol
        matches = np.absolute(delta).max() < 0.01 # extra relaxed
        if not matches:
            warnings.warn("reconstructed model spectrum does not match pPXF "
                          "output model - quartiles over pixels of the "
                          "fractional deviations are: {:.2e} {:.2e} {:.2e} "
                          "{:.2e}".format(*utl.quartiles(delta)))
        return proc_outputs

    def run_fit(self, crop_factor=5): 
        """
        Perform the actual pPXF fit, along with processing of the fit
        output and Monte Carlo fits to determine errors. This driver
        ensures that the data and templates are properly prepared.

        Fit outputs are not returned, but stored in six pPXFdriver
        attributes. See init_output_containers for details.
        """
        if self.fit_complete:
            warnings.warn("A pPXF fit to this set of spectra has "
                          "already been computed - overwriting...")
        # determine size of ir-convolved templates
        all_ir = self.specset.metaspectra["ir"]
        edge_ir = all_ir[:, [0, -1]].max(axis=0) # max over templates
        edge_sigma = edge_ir/const.gaussian_fwhm_over_sigma
        self.valid_temp_range = (self.templib.spectrumset.spec_region +
                                 crop_factor*edge_sigma*np.array([1, -1]))
        first_spectrum = self.specset.get_subset([self.specset.ids[0]])
        test_matched_library = self.prepare_library(first_spectrum)
        self.num_temp_samples = test_matched_library.spectrumset.num_samples
        self.num_temps = test_matched_library.spectrumset.num_spectra
        # prep for fit
        self.init_output_containers()
        self.bestfit_output["waves"] = self.specset.waves
        # run actual fit
        for spec_iter, target_id in enumerate(self.specset.ids):
            print ("fitting spectrum {} ({} of {})..."
                   "".format(target_id, spec_iter + 1,
                             self.specset.num_spectra))
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
            fitter = ppxf.ppxf(library_spectra_cols, target_spec.spectra[0],
                               target_spec.metaspectra["noise"][0],
                               self.velscale, self.initial_gh,
                               goodpixels=good_pix_indices,
                               vsyst=velocity_offset, plot=False,
                               quiet=True, **self.ppxf_kwargs)
            bestfit_output = self.get_raw_pPXF_results(fitter)
            proc_results = self.process_pPXF_results(fitter, self.VEL_FACTOR)
            bestfit_output.update(proc_results)
            bestfit_output["binid"] = target_id
            utl.fill_dict(self.bestfit_output, bestfit_output, spec_iter)
            self.save_matched_templates(matched_library, spec_iter)
            # construct error estimate from Monte Carlo trial spectra
            if not (self.num_trials > 1):
                continue
            base = self.bestfit_output["best_model"][spec_iter, :]
            chisq_dof = self.bestfit_output["chisq_dof"][spec_iter]
            noise_scale = target_spec.metaspectra["noise"][0]
            noise_scale *= np.sqrt(chisq_dof)
                # this sets the noise level for the Monte Carlo trials to be
                # roughly the size of the actual fit residuals - i.e., we
                # assume the model is valid and that a high/low chisq per dof
                # is due to under/overestimated errors
            noise_draw = np.random.randn(self.num_trials, *base.shape)
                # uniform, uncorrelated Gaussian pixel noise
            trial_spectra = base + noise_draw*noise_scale
            base_gh_params = self.bestfit_output["gh_params"][spec_iter, :]
            self.mc_output["waves"] = self.specset.waves
            self.mc_output["binid"][spec_iter] = target_id
            for trial_iter, trial_spectrum in enumerate(trial_spectra):
                trial_fitter = ppxf.ppxf(library_spectra_cols,
                                         trial_spectrum, noise_scale,
                                         self.velscale, base_gh_params,
                                         goodpixels=good_pix_indices,
                                         vsyst=velocity_offset, plot=False,
                                         quiet=True, **self.ppxf_kwargs)
                fit_index = (spec_iter, trial_iter)
                self.mc_output["noiselevels"][spec_iter, :] = noise_scale
                self.mc_output["spectrum"][fit_index] = trial_spectrum
                mc_output = self.get_raw_pPXF_results(trial_fitter)
                proc_mc_output = self.process_pPXF_results(trial_fitter,
                                                           self.VEL_FACTOR)
                mc_output.update(proc_mc_output)
                utl.fill_dict(self.mc_output, mc_output, fit_index)
        return

    def write_outputs(self,paths_dict):
        """
        Write driver outputs to file.
        Place all outputs in destination_dir, with run_name as 
          base filename.
        All results are packaged into one organized .fits file for
          the main output, and one similarly organized .fits file
          for the mc output.
        If only one bin was fit, output an additional text file to
          store list of nonzero template spectra in a convenient
          form. (This is used when fitting the full galaxy spectrum
          with the full Miles library, so the list can be used as
          input to further fits.)
        """
        #Set up the header with relevant metadata
        baseheader = fits.Header()
        for name, [ppxf_name, func] in self.PPXF_REQUIRED_INPUTS.iteritems():
            if len(name) > 8:
                name = name[-8:]
            baseheader.append((name, self.ppxf_kwargs[ppxf_name]))
        baseheader.append(("velscale", self.velscale,
                           "spectral velocity step used for pPXF [km/s]"))
        gauss_names = ['vel', 'sigma']
        number_of_h_params = self.ppxf_kwargs["moments"] - 2
        h_names = ["h" + str(n) for n in xrange(3, 3 + number_of_h_params)] 
        param_names = gauss_names + h_names
        param_units = ['km/s', 'km/s'] + ["1"]*number_of_h_params
        for param_iter, param_name in enumerate(param_names):
            init_value = self.initial_gh[param_iter]
            unit = param_units[param_iter]
            baseheader.append(("{}_0".format(param_name), init_value,
                               "starting value of {} [{}]"
                               "".format(param_name, unit)))
        baseheader.append(("srcfile", self.sourcefile, "source file"))
        baseheader.append(("srcdate", self.sourcedate, "source file date"))

        # save main fits file, looping over output containers automatically
        hdulist = []
        primaryhdu = True
        for hduname in self.output_hdunames:
            header = baseheader.copy()
            outputnames = self.output_names[hduname]
            if len(outputnames)==1:
                data = self.bestfit_output[outputnames[0]]
            else:
                data = [self.bestfit_output[name] for name in outputnames]
            header.append(("columns", ",".join(outputnames)))
            if primaryhdu:
                header.append(("primary",hduname))
                hdu = fits.PrimaryHDU(data=data,header=header)
                primaryhdu = False
            else:
                hdu = fits.ImageHDU(data=data,header=header,name=hduname)
            hdulist.append(hdu)
        fitshdulist = fits.HDUList(hdus=hdulist)
        fitshdulist.writeto(paths_dict['main'], clobber=True)

        if self.num_trials == 0:
            print 'No MC runs done, saving only main fits file.'
            return

        # save mc fits file, looping over output containers automatically
        hdulist = []
        primaryhdu = True
        for hduname in self.mcoutput_hdunames:
            header = baseheader.copy()
            outputnames = self.mcoutput_names[hduname]
            if len(outputnames)==1:
                data = self.mc_output[outputnames[0]]
            else:
                data = [self.mc_output[name] for name in outputnames]
            header.append(("columns", ",".join(outputnames)))
            if primaryhdu:
                header.append(("primary",hduname))
                hdu = fits.PrimaryHDU(data=data,header=header)
                primaryhdu = False
            else:
                hdu = fits.ImageHDU(data=data,header=header,name=hduname)
            hdulist.append(hdu)
        fitshdulist = fits.HDUList(hdus=hdulist)
        fitshdulist.writeto(paths_dict['mc'], clobber=True)

        return
