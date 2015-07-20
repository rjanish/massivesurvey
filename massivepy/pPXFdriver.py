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
                 initial_gh=None, num_trials=None, **ppxf_kwargs):
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

    def init_output_containers(self):
        """
        Prepare containers for all pPXF outputs
        """
        num_spectra = self.specset.num_spectra
        num_samples = self.specset.num_samples
        num_trials = self.num_trials
        num_add_weights = self.ppxf_kwargs['degree'] + 1  # deg 0 --> 1 const
        num_mul_weights = self.ppxf_kwargs['mdegree'] + 1  
        # convolved template libraries
        temp_lib_shape = [num_spectra, self.num_temps, self.num_temp_samples]
        self.matched_templates = {"spectra":  np.zeros(temp_lib_shape),
                                  "ir":  np.zeros(temp_lib_shape),
                                  "waves":  np.zeros(self.num_temp_samples)}
        # pPXF outputs
        output_shapes = {
            "best_model": [num_samples],
            "gh_params": [self.ppxf_kwargs['moments']],
            "chisq_dof": [1],
            "template_weights": [self.num_temps],
            "add_weights": [num_add_weights],
            "mul_weights": [num_mul_weights],
            "unscaled_lsq_errors": [self.ppxf_kwargs['moments']],
                # error estimate from least-square fit's covariance matrix
            "scaled_lsq_errors": [self.ppxf_kwargs['moments']],
                # least-square error estimate scaled by best-fit chi-squared
            "mul_poly": [num_samples],
            "add_poly": [num_samples],
            "smoothed_temps": [self.num_temps, self.num_temp_samples],
                # templates smoothed by losvd
            "model_temps": [self.num_temps, num_samples],
                # smoothed templates also scaled by multiplicative polynomial
            "model_temps_fluxes": [self.num_temps],
            "add_fluxweights": [num_add_weights],
                # weighted normalized to equal fractional flux level
            "template_fluxweights": [self.num_temps]}
        self.bestfit_output = {}
        self.mc_output = {}
        for output, shape in output_shapes.iteritems():
            self.bestfit_output[output] = np.zeros([num_spectra] + shape)
            self.mc_output[output] = np.zeros([num_spectra, num_trials]
                                              + shape)
        # trial noises and spectra
        self.mc_inputs = {
            "noiselevels": np.zeros((num_spectra, num_samples)),
            "spectra": np.zeros((num_spectra, num_trials, num_samples))}
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
        matches = np.absolute(delta).max() < const.relaxed_tol
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
        attributes. The first of these store the results of the fits
        to the passed spectra:
          main_input - record of inputs to pPXF
          main_rawoutput - record of direct pPXF outputs
          main_procoutput - processed pPXF outputs
        and the next three hold the corresponding results for the Monte
        Carlo fits, one set of trial fits for each input spectrum:
          mc_input - record of inputs to pPXF
          mc_rawoutput - record of direct pPXF outputs
          mc_procoutput - processed pPXF outputs
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
            for trial_iter, trial_spectrum in enumerate(trial_spectra):
                trial_fitter = ppxf.ppxf(library_spectra_cols,
                                         trial_spectrum, noise_scale,
                                         self.velscale, base_gh_params,
                                         goodpixels=good_pix_indices,
                                         vsyst=velocity_offset, plot=False,
                                         quiet=True, **self.ppxf_kwargs)
                fit_index = (spec_iter, trial_iter)
                self.mc_inputs["noiselevels"][spec_iter, :] = noise_scale
                self.mc_inputs["spectra"][fit_index] = trial_spectrum
                mc_output = self.get_raw_pPXF_results(trial_fitter)
                utl.fill_dict(self.mc_output, mc_output, fit_index)
        return

    def write_outputs(self,paths_dict,debug_mc=False):
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
        #Note: all contents from main_input, main_rawoutput, main_procoutput
        # are either saved to the .fits file or checked for being zero or
        # none EXCEPT main_procoutput['smoothed_temps'], because they are
        # extremely inconvenient, large, and probably not useful.
        
        #Set up the main run fits file
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
        #Now do stuff that does not match between bins
        #HDU 1: gh moments and lsq errors
        ###Add option to save gh_params as text file?###
        gh_params = self.bestfit_output['gh_params']
        lsqerr = self.bestfit_output['unscaled_lsq_errors']
        scaledlsq = self.bestfit_output['scaled_lsq_errors']
        moment_info = [gh_params,lsqerr,scaledlsq]
        moment_info_columns = "ghparams,lsqerr,scaledlsq"
        header_gh = baseheader.copy()
        header_gh.append(("axis1", "moment"))
        header_gh.append(("axis2", "bin"))
        header_gh.append(("axis3", moment_info_columns))
        header_gh.append(("primary","gh_moments"))
        hdu_gh = fits.PrimaryHDU(data=moment_info,
                                 header=header_gh)
        #HDU 2: templates and weights (raw and flux-weighted)
        t_ids = [self.templib.spectrumset.ids]*self.specset.num_spectra
        t_weights = self.bestfit_output['template_weights']
        t_mflux = self.bestfit_output['model_temps_fluxes']
        t_fw = self.bestfit_output['template_fluxweights']
        template_info = [t_ids,t_weights,t_mflux,t_fw]
        template_info_columns = "id,weight,modelflux,fluxweight"
        header_temps = baseheader.copy()
        header_temps.append(("axis1", "template"))
        header_temps.append(("axis2", "bin"))
        header_temps.append(("axis3", template_info_columns))
        hdu_temps = fits.ImageHDU(data=template_info,
                                     header=header_temps,
                                     name='template_info')
        #Now the text file for the templates
        if len(t_weights)==1:
            #Collapse extraneous dimension for bin number, convert to 2d array
            t_array = np.array([info[0] for info in template_info]).T
            #Get rid of zero weights (weights should be in second column)
            ii = np.nonzero(t_array[:,1])[0]
            t_array_nonzero = t_array[ii,:]
            #Format first column (id number) as int
            fmt = ['%i']
            fmt.extend(['%-8g']*(len(template_info)-1))
            np.savetxt(paths_dict['temps'],
                       t_array_nonzero,
                       header='columns are {}'.format(template_info_columns),
                       fmt=fmt,delimiter='\t')
        #HDU 3: spectrum and other related things (per bin)
        best_model = self.bestfit_output['best_model']
        mul_poly = self.bestfit_output['mul_poly']
        add_poly = self.bestfit_output['add_poly']
        spec_info = [best_model,mul_poly,add_poly]
        spec_info_columns = "bestmodel,mulpoly,addpoly"
        header_spec = baseheader.copy()
        header_spec.append(("axis1", "pixel"))
        header_spec.append(("axis2", "bin"))
        header_spec.append(("axis3", spec_info_columns))
        hdu_spec = fits.ImageHDU(data=spec_info,
                                 header=header_spec,
                                 name='spectrum_info')

        #HDU 4: anything that goes one-number-per-bin
        binids = self.specset.ids
        chisq_dof = self.bestfit_output['chisq_dof'].reshape(binids.shape)
        bins_info = [binids,chisq_dof]
        bins_info_columns = "binid,chisq"
        header_bins = baseheader.copy()
        header_bins.append(("axis1", "bin"))
        header_bins.append(("axis2", bins_info_columns))
        hdu_bins = fits.ImageHDU(data=bins_info,
                                header=header_bins,
                                name='bin_info')

        #HDU 5: The add_weights, mul_weights
        nbins = len(self.bestfit_output['add_weights'])
        addmul_info = []
        for i in range(nbins):
            add_weights = list(self.bestfit_output['add_weights'][i])
            mul_weights = list(self.bestfit_output['mul_weights'][i])
            flux_add_weights = list(self.bestfit_output['add_fluxweights'][i])
            addmul_info.append(add_weights + flux_add_weights + mul_weights)
        addmul_info_columns = "{}addweights,fluxaddweights,{}mulweights".format(len(add_weights),len(mul_weights))
        header_addmul = baseheader.copy()
        header_addmul.append(("axis1", addmul_info_columns))
        header_addmul.append(("axis2", "bin"))
        hdu_addmul = fits.ImageHDU(data=addmul_info,
                                header=header_addmul,
                                name='add_mul_weights')

        #HDU 6: The model templates
        header_mtemps = baseheader.copy()
        header_mtemps.append(("axis1", "pixel"))
        header_mtemps.append(("axis2", "template"))
        hdu_mtemps = fits.ImageHDU(data=self.bestfit_output['model_temps'],
                                   header=header_mtemps,
                                   name='model_temps')

        #HDU 7: The wavelengths (same for all bins, so not in spectrum hdu)
        header_waves = baseheader.copy()
        hdu_waves = fits.ImageHDU(data=self.specset.waves,
                                  header=header_waves,
                                  name='wavelengths')

        #Now collect all the HDUs for the fits file
        hdu_all = fits.HDUList(hdus=[hdu_gh,hdu_temps,hdu_spec,hdu_bins,
                                     hdu_addmul,hdu_mtemps,hdu_waves])
        hdu_all.writeto(paths_dict['main'], clobber=True)

        #Now do everything again for the mc runs. Only save things that
        # actually change by run, i.e. input spectrum and all of the outputs
        #If the number of runs is not more than one, the mc portion is
        # skipped, so here just return without saving mc stuff if it does not
        # exist. (in that case output dicts exist but are empty, no keys)
        if self.num_trials == 0:
            print 'No MC runs done, saving only main fits file.'
            return
        mc_baseheader = fits.Header()
        #HDU 1
        mc_gh_params = self.mc_output['gh_params']
        mc_lsqerr = self.mc_output['unscaled_lsq_errors']
        mc_scaledlsq = self.mc_output['scaled_lsq_errors']
        mc_moment_info = [mc_gh_params,mc_lsqerr,mc_scaledlsq]
        mc_moment_info_columns = "ghparams,lsqerr,scaledlsq"
        mc_header_gh = mc_baseheader.copy()
        mc_header_gh.append(("axis1", "moment"))
        mc_header_gh.append(("axis2", "mcrun"))
        mc_header_gh.append(("axis3", "bin"))
        mc_header_gh.append(("axis4", mc_moment_info_columns))
        mc_header_gh.append(("primary", "gh_moments"))
        mc_hdu_gh = fits.PrimaryHDU(data=mc_moment_info,
                                    header=mc_header_gh)
        #HDU 2
        mc_t_weights = self.mc_output['template_weights']
        mc_template_info = [mc_t_weights]
        mc_template_info_columns = "id,weight"
        mc_header_temps = mc_baseheader.copy()
        mc_header_temps.append(("axis1", "template"))
        mc_header_temps.append(("axis2", "mcrun"))
        mc_header_temps.append(("axis3", "bin"))
        mc_header_temps.append(("axis4", mc_template_info_columns))
        mc_hdu_temps = fits.ImageHDU(data=mc_template_info,
                                     header=mc_header_temps,
                                     name='template_info')
        #HDU 3
        mc_spectrum = self.mc_inputs['spectra']
        mc_best_model = self.mc_output['best_model']
        mc_mul_poly = self.mc_output['mul_poly']
        mc_add_poly = self.mc_output['add_poly']
        mc_spec_info = [mc_spectrum,mc_best_model,mc_mul_poly,mc_add_poly]
        mc_spec_info_columns = "spec,bestmodel,mulpoly,addpoly"
        mc_header_spec = mc_baseheader.copy()
        mc_header_spec.append(("axis1", "pixel"))
        mc_header_spec.append(("axis2", "bin"))
        mc_header_spec.append(("axis3", mc_spec_info_columns))
        mc_hdu_spec = fits.ImageHDU(data=mc_spec_info,
                                    header=mc_header_spec,
                                    name='spectrum_info')
        #Now collect all the HDUs for the fits file
        if debug_mc:
            mc_hdu_all = [mc_hdu_gh,mc_hdu_temps,mc_hdu_spec]
        else:
            mc_hdu_all = [mc_hdu_gh]
        mc_fits = fits.HDUList(hdus=mc_hdu_all)
        mc_fits.writeto(paths_dict['mc'], clobber=True)

        ###
        #Notes about what I am saving of the mc runs!
        #   -from self.mc_input: spectrum
        #   -from self.mc_rawoutput: gh_params, template_weights, 
        #    unscaled_lsq_errors, best_model
        #   -from self.mc_procoutput: model_temps_fluxes, scaled_lsq_errors,
        #    mul_poly, flux_template_weights, add_poly
        #Everything else is left out. If debug_mc=False, everything but the
        # moment stuff is left out.
        ###
        return
