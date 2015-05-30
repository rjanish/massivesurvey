"""
This module handles the storage and common calculation using
collections of spectra.
"""


import collections as collect
import warnings
import os

import numpy as np
import scipy.integrate as integ
import scipy.interpolate as inter
import astropy.units as units
import astropy.io.fits as fits

import utilities as utl
import massivepy.constants as const


class SpectrumSet(object):
    """
    This class holds a set of spectra along with their metadata.

    The metadata are the focus here - this class mainly serves to
    enforce that all of the information needed to interpret spectra is
    explicitly recorded along with the spectral data. The included
    methods for I/O and manipulation of spectra are meant to enforce
    the preservation and automatic updating of the metadata as needed.
    """
    def __init__(self, spectra=None, bad_data=None, noise=None,
                 ir=None, spectra_ids=None, wavelengths=None,
                 spectra_unit=None, wavelength_unit=None,
                 comments={}, name=None):
        """
        Mandatory arguments here force explicit recording of metadata.
        When this function returns, the object will hold all of the
        spectra metadata relevant for kinematic modeling. Input array
        data is copied.

        Args:
        spectra - 1d or 2d arraylike
            The actual spectrum values. Each spectrum occupies a row,
            and are assumed to be identical in sampling and units. A
            single 1d array may be passed, which is interpreted as a
            set containing a single spectrum and is converted as such.
        bad_data - boolean arraylike, matches the shape of spectrum
            Indicates the location of junk data that is to be ignored
            in computations. True: junk, False: valid data.
            assumed to match those of spectrum.
        noise - arraylike, matches the shape of spectrum
            Estimate of noise level in spectrum, assumed to be the
            half-width of a 1-sigma confidence interval. The units are
            assumed to match those of spectrum.
        ir - arraylike, matches the shape of spectrum
            The spectral resolution of the instrument that recorded
            the spectra. This is assumed to be given as a Gaussian
            FWHM, with the units matching wavelengths.
        spectra_ids - 1d int arraylike, size matches number of spectra
            Unique identifiers labeling the spectra in the order given
            in the above 'spectra' argument. Assumed to be integers.
        wavelengths - 1d arraylike
            Wavelength sample values, the same for each spectrum.
        spectra_unit - astropy unit-like
            The unit in which the values of spectra are given.
        wavelength_unit - astropy unit-like
            The unit in which the values of wavelengths are given.
            This is assumed to be some unit of length.
        comments - dict, default is empty
            This is a dictionary to store comments about the spectra.
            The keys are treated as strings, otherwise there are no
            formatting restrictions. These comments will be shuffled
            along with I/O operations in file headers.
        name - str, default is 'None'
            The name of this set of spectra
        """
        # check spectra format
        self.spectra = np.array(spectra, dtype=float)
        if self.spectra.ndim not in [1, 2]:
            raise ValueError("Invalid spectra shape: {}. Must have 1 or 2 "
                             "dimensions.".format(self.spectra.shape))
        self.spectra = np.atleast_2d(self.spectra)  # spectrum shape now 2D
        self.num_spectra, self.num_samples = self.spectra.shape
        # check ids format
        self.ids = np.asarray(spectra_ids, dtype=int)
        if ((self.ids.ndim != 1) or
            (self.ids.size != self.num_spectra)):
            raise ValueError("Invalid spectra ids shape: {}. Must be "
                             "1D and match the number of spectra: {}."
                             "".format(self.ids.shape, self.num_spectra))
        count_ids = collect.Counter(self.ids) # dict: id:num_of_appearences
        duplicates = [id for id, count in count_ids.iteritems() if count > 1]
        num_duplicates = len(duplicates)
        if num_duplicates != 0:
            raise ValueError("Invalid spectra ids - ids are not unique: {} "
                             "each appear more than once".format(duplicates))
        # check waves format
        self.waves = np.array(wavelengths, dtype=float)
        if ((self.waves.ndim != 1) or
            (self.waves.size != self.num_samples)):
            raise ValueError("Invalid wavelength shape: {}. Must be "
                             "1D and match the size of spectra: {}."
                             "".format(self.waves.shape, self.num_samples))
        self.spec_region = utl.min_max(self.waves)
        # check metaspectra format
        # 'metaspectra' are those metadata that have a spectra-like form
        metaspectra_inputs = {"noise":noise, "ir":ir, "bad_data":bad_data}
        float_2d = lambda a: np.atleast_2d(np.asarray(a, dtype=float))
        bool_2d = lambda a: np.atleast_2d(np.asarray(a, dtype=bool))
        conversions = {"noise":float_2d, "ir":float_2d, "bad_data":bool_2d}
        self.metaspectra = {key:conversions[key](data)
                            for key, data in metaspectra_inputs.iteritems()}
            # metaspectra now float or bool valued and have dimension >= 2
        for key, data in self.metaspectra.iteritems():
            if data.shape != self.spectra.shape:
                error_msg = ("Invalid {} shape: {}. "
                             "Must match the shape of spectra: {}."
                             "".format(key, data.shape, self.spectra.shape))
                raise ValueError(error_msg)
        # remaining arg checks
        self.spec_unit = units.Unit(spectra_unit)
        wavelength_unit.to(units.cm)  # check if wavelength_unit is a length
        self.wave_unit = units.Unit(wavelength_unit)
        self.integratedflux_unit = self.spec_unit*self.wave_unit
        self.comments = {str(k):v for k, v in comments.iteritems()}
        self.name = str(name)

    def get_subset(self, ids, get_selector=False):
        """
        Extract subset of spectral data with the passed spectrum ids.
        Ordering of spectra is NOT preserved.

        Returns a new spectrumset object containing a subset of the
        spectra and associated metadata from the original spectrumset.
        Metadata that is by-assumption uniform over all spectra, such
        as the wavelengths, spectral unit, etc., is also preserved.

        Selecting of spectra is NOT done by their storage index within
        the spectrumset object, but by the values of the spectra ids:
        > specset = spectrumset(...)
        > numbers = [5, 10, 11, 19]
        > specset.get_subset(numbers)
        returns a spectrumset containing all spectra with id values of
        5, 10, 11, or 19. I.e.,
        > ids_selector = np.in1d(specset.ids, numbers)
        > subset_via_guts = specset.spectra[np.in1d(specset.ids, numbers)]
        > subset_via_method = specset.get_subset(numbers).spectra
        > np.all(subset_via_guts == subset_via_method)
        will return True.

        ids can be given in any shape, but are treated as 1D. Each id
        given must match exactly one id in the current spectrumset,
        and the passed ids must not have duplicates. This is to prevent
        accidental missing and mixed-up spectra - if such uncommon
        combinations are needed they must be build explicitly.

        If get_selector is True, also returns boolean index array.
        """
        # check for passed duplicate
        ids_wanted = np.asarray(ids, dtype=int).flatten()
        count_ids = collect.Counter(ids_wanted) # dict: id:num_of_appearences
        duplicates = [id for id, count in count_ids.iteritems() if count > 1]
        num_duplicates = len(duplicates)
        if num_duplicates != 0:
            raise ValueError("Invalid spectra id - ids are not unique: {} "
                             "each appear more than once".format(duplicates))
        # generate index - check for unmatched ids
        index = np.zeros(self.num_spectra, dtype=bool)
        for id in ids_wanted:
            selector = (self.ids == id)
            num_selected = selector.sum()
            if num_selected != 1:
                raise ValueError("Invalid spectra id: {} matched {} "
                                 "spectra in set".format(id, num_selected))
            else:
                index = index | selector
        subset = SpectrumSet(spectra=self.spectra[index, :],
                             bad_data=self.metaspectra["bad_data"][index, :],
                             noise=self.metaspectra["noise"][index, :],
                             ir=self.metaspectra["ir"][index, :],
                             spectra_ids=self.ids[index],
                             wavelengths=self.waves,
                             spectra_unit=self.spec_unit,
                             wavelength_unit=self.wave_unit,
                             comments=self.comments)
        if get_selector:
            return subset, index
        else:
            return subset

    def get_masked(self, name=None):
        """ Return a masked array version of passed data """
        mask = self.metaspectra['bad_data']
        if name == 'spectra':
            masked = np.ma.array(self.spectra, mask=mask, fill_value=np.nan)
        elif name == 'waves':
            masked = np.ma.array(self.waves, mask=mask, fill_value=np.nan)
        else:
            masked = np.ma.array(self.metaspectra[name],
                                 mask=mask, fill_value=np.nan)
        return masked

    def get_wavescale(self):
        """ Return the wavelength step, whining if not linear spaced """
        delta = self.waves[1:] - self.waves[:-1]
        wavescale = delta[0]
        residual = np.absolute(delta - wavescale).max()
        is_linear_sampled = residual < const.float_tol
        if not is_linear_sampled:
            raise ValueError("spectra are not linear-sampled")
        return wavescale

    def is_linear_sampled(self):
        """ Check if wavelengths are linear spaced. Boolean output. """
        try:
            self.get_wavescale()
            return True
        except ValueError:
            return False

    def get_logscale(self):
        """ Return the log-wavelength step, whining if not log spaced """
        log_waves = np.log(self.waves)
        delta = log_waves[1:] - log_waves[:-1]
        logscale = delta[0]
        residual = np.absolute(delta - logscale).max()
        is_log_sampled = residual < const.float_tol
        if not is_log_sampled:
            raise ValueError("spectra are not log-sampled")
        return logscale

    def is_log_sampled(self):
         """ Check if wavelengths are log spaced. Boolean output. """
        try:
            self.get_logscale()
            return True
        except ValueError:
            return False

    def resample(self, new_waves):
        """
        Re-sample spectra to have a new wavelength sampling.

        The only restriction on the new sampling is that all points
        must lie within the previous spectral range. New spectral
        values are determined by linear interpolation. Spectra and
        metadata will be updated in-place.

        Args:
        new_waves - 1d arraylike
            The new wavelength values on which to sample the spectra.
            Units are assumed to match the previous wavelength units.
        """
        new_num_samples = new_waves.shape[0]
        new_spec_shape = (self.num_spectra, new_num_samples)
        new_spectra = np.zeros(new_spec_shape)
        new_metaspectra = {}
        for name in self.metaspectra:
            new_metaspectra[name] = np.zeros(new_spec_shape)
        for spec_index in xrange(self.num_spectra):
            spec_func = inter.interp1d(self.waves, self.spectra[spec_index])
            new_spectra[spec_index] = spec_func(new_waves)
            for name, mspec in self.metaspectra.iteritems():
                mspec_func = inter.interp1d(self.waves, mspec[spec_index])
                new_mspec_values = mspec_func(new_waves)
                if name == 'bad_data':
                    new_mspec_values = new_mspec_values.astype(bool)
                        # re-sampled data is valid only if the nearest
                        # bracketing old-sampling values are both valid
                new_metaspectra[name][spec_index] = new_mspec_values
        self.spectra = new_spectra
        self.metaspectra = new_metaspectra
        self.waves = new_waves
        self.num_samples = new_num_samples
        self.spec_region = np.asarray([new_waves.min(), new_waves.max()])
        return

    def log_resample(self, target_logscale=None):
        """
        Re-sample spectra to have logarithmic spacing.

        The re-sampling can be done either with a given logscale, or
        by preserving the number of sample points. The spectral region
        will be preserved in either case (save for a small inward
        shift due to roundoff).

        The logscale used here is defined to be:
        $ \log(\lambda_{n + 1}) - \log(\lambda{n}) = logscale $

        New spectra and metadata values are computed by resample.

        Args:
        target_logscale - float, default=None
            If given, spectra will be re-sampled using the passed
            logscale. If not specified, re-sampling will instead
            preserve the number of data points.

        Return: logscale
        logscale - float
            The logscale of the now re-sampled spectrum
        """
        if self.is_log_sampled():
            if target_logscale is None:
                warnings.warn("spectra is already log-spaced - aborting "
                              "log_resample since no target logscale given")
                return
            current_logscale = self.get_logscale()
            logscale_match = np.abolute(target_logscale -
                                        current_logscale) < const.float_tol
            if logscale_match:
                warnings.warn("spectra already has desired "
                              "logscale, aborting log_resample")
                return
        # re-sampling is required
        inward_scaling = 1 + np.array([1, -1])*const.float_tol
        new_spec_region = self.spec_region*inward_scaling
            # prevents unintended extrapolation due to roundoff
        log_ends = np.log(new_spec_region)
        if target_logscale is None:  # preserve sample number
            log_w = np.linspace(log_ends[0], log_ends[1], self.num_samples)
            target_logscale = log_w[1] - log_w[0]
        else:  # fix log-spacing
            log_w = np.arange(log_ends[0], log_ends[1], target_logscale)
        new_waves = np.exp(log_w)
        self.resample(new_waves)
        return

    def linear_resample(self, step=None):
        """
        Re-sample spectra to have linear spacing.

        The re-sampling can be done either with a given step size, or
        by preserving the number of sample points. The spectral region
        will be preserved in either case (save for a small inward
        shift due to roundoff).

        The step used here is defined to be:
        $ \lambda_{n + 1} - \lambda{n} = step $

        New spectra and metadata values are computed by resample.
        
        Args:
        step - float, default=None
            If given, spectra will be re-sampled using the passed
            step. If not specified, re-sampling will instead
            preserve the number of data points.

        Return: step
        step - float
            The step of the now re-sampled spectrum
        """
        if self.is_linear_sampled():
            raise ValueError("Spectrum is already linear-sampled.")
        inward_scaling = 1 + np.array([1, -1])*const.float_tol
        new_ends = self.spec_region*inward_scaling
            # prevents unintended extrapolation due to roundoff
        if step is None:  # preserve sample number
            new_waves = np.linspace(new_ends[0], new_ends[1],
                                    self.num_samples)
            step = new_waves[1] - new_waves[0]
        else:  # fix step size
            new_waves = np.arange(new_ends[0], new_ends[1], step)
        self.resample(new_waves)
        return step

    def compute_flux(self, interval=None, ids=None):
        """
        Compute the flux of spectrum over the given interval.

        The flux is computed as the integral of the spectrum over
        wavelength. This is done using whatever units are given at
        the time of class construction, so the output of this routine
        is not necessarily expressed in units of flux. The integration
        is done using Simpson's quadrature.

        Args:
        interval - 1D, 2-element arraylike; default = full data range
            The wavelength interval over which to compute the flux,
            expressed as an array [lamba_start, lamba_end]. This
            interval must be contained in the data's spectral range.
        ids - 1D int arraylike, default is all
            The id's of the spectra for which to compute the flux. By
            default, the flux of all spectra will be computed.

        Return:
        fluxes - arraylike
            The wavelength-integrated flux of each spectrum.
        """
        if ids is None:
            ids = self.ids
        else:
            ids = np.asarray(ids, dtype=int)
        if interval is None:
            inward_scaling = 1 + np.array([1, -1])*const.float_tol
            interval = self.spec_region*inward_scaling
                # shift to avoid roundoff in comparisons with full interval
        elif not utl.interval_contains_interval(self.spec_region, interval):
            raise ValueError("Invalid interval: {}. Region must be "
                             "contained in data spectral range: {}."
                             "".format(interval, self.spec_region))
        flux_region = utl.in_linear_interval(self.waves, interval)
        fluxes = np.zeros(ids.size)
        for result_index, id in enumerate(ids):
            spec_index = (self.ids == id)
            valid_data = ~self.metaspectra["bad_data"][spec_index]
            valid_data = valid_data.reshape(valid_data.size)
                # indexing by 1d bool array defaults to output shape (1, N),
                # convert to 1d output shape (N,) needed for indexing below
            to_integrate = flux_region & valid_data
            flux = integ.simps(self.spectra[spec_index, to_integrate],
                               self.waves[to_integrate])
            fluxes[result_index] = flux
        return fluxes

    def compute_mean_s2n(self, ids=None):
        """
        Return the mean S/N over wavelength for the spectra with the
        passed ids, or by default all of the spectra.
        """
        if ids is None:
            ids = self.ids
        else:
            ids = np.asarray(ids, dtype=int)
        index = np.in1d(self.ids, ids)
        masked_spectra = self.get_masked('spectra')
        masked_noise = self.get_masked('noise')
        s2n = masked_spectra[index, :]/masked_noise[index, :]
        mean_s2n = np.mean(s2n, axis=1)  # new ma array - default fill value
        mean_s2n.set_fill_value(np.nan)
        s2n_values = mean_s2n.filled() # gets data with filling
        return s2n_values

    def get_normalized(self, norm_func=None, norm_value=1):
        """
        Normalize spectral data, returning as a new SpectrumSet.
        Metadata are scaled consistent with the corresponding spectra.

        The norm type is controlled by the passed function norm_func
        and norm_value - the norm will be set such that all spectra
        have norm_func(spectrum) = norm_value. It is assumed that
        norm_func is linear: norm_func(v*M) = v*norm_func(M).
        """
        current_values = norm_func(self)
        norm_factors = norm_value/current_values
        normed_spectra = (self.spectra.T*norm_factors).T
            # mult each row of spectra by corresponding entry in norm_factors
        normed_noise = (self.metaspectra["noise"].T*norm_factors).T
        extened_comments = self.comments.copy()
        extened_comments["Normalization"] = ("Normalized to have unit {}"
                                             "".format(norm_func.__name__))
        return SpectrumSet(spectra=normed_spectra, noise=normed_noise,
                           bad_data=self.metaspectra["bad_data"],
                           ir=self.metaspectra["ir"], spectra_ids=self.ids,
                           wavelengths=self.waves,
                           spectra_unit=self.spec_unit,
                           wavelength_unit=self.wave_unit,
                           comments=extened_comments)

    def collapse(self, weight_func=None, id=None,
                 norm_func=None, norm_value=None):
        """
        Combine all spectra into a single spectrum, treating metadata
        consistently, returned as a new SpectrumSet object.

        Each spectrum is first normalized such that all have equal
        equal over the full spectral range. Weights for each spectrum
        in the combination are determined by the passed weight_func
        acting on the normalized spectra - weight_func must accept a
        SpectrumSet and return an array of combination weights. The
        combination is done via a clipped mean.
        """
        delta = self.spec_region[1] - self.spec_region[0]
        fluxnormed_set = self.get_normalized(norm_func, norm_value)
            # normalized by flux, with spectra numerical values ~ 1.0
        # spec_median = lambda s: np.median(s.get_masked('spectra'), axis=1)
        # fluxnormed_set = self.get_normalized(spec_median, 1)
        weight = weight_func(fluxnormed_set)
        if weight.ndim == 1:
            weight = np.vstack((weight,)*fluxnormed_set.num_samples).T
                # array with each row a constant value
        comb = utl.clipped_mean(fluxnormed_set.spectra, weights=weight,
                                noise=fluxnormed_set.metaspectra['noise'],
                                mask=fluxnormed_set.metaspectra['bad_data'])
        comb_spectra, comb_noise, comb_bad_data, clipped = comb
        total_weight = weight.sum(axis=0)
        comb_ir = (self.metaspectra["ir"]*weight).sum(axis=0)/total_weight
        extened_comments = self.comments.copy()
        extened_comments["Binning"] = ("This spectrum was binned from {} "
                                       "spectra {} with weight function {}"
                                       "".format(self, self.ids,
                                                 weight_func.__name__))
        return SpectrumSet(spectra=comb_spectra, bad_data=comb_bad_data,
                           noise=comb_noise, ir=comb_ir, spectra_ids=[id],
                           wavelengths=self.waves, comments=extened_comments,
                           spectra_unit=self.spec_unit,
                           wavelength_unit=self.wave_unit)

    def gaussian_convolve(self, std, crop_factor=5):
        """
        Convolve each spectrum with a centered Gaussian having the
        passed standard deviation, which may be a function of
        wavelength. A new SpectrumSet will be returned, with updated
        spectra and resolution.

        Args:
        std - 1d arraylike
            The standard deviation of the Gaussian smoothing kernel.
            The units are assumed to be the SpectrumSet's wavelength
            units, and the wavelength sampling must match spectra.
        crop_factor - float, default=5
            To eliminate edge effects, the output will be cropped to
            remove all data within sigma*crop_factor of the edge.

        Returns: specset
        specset - SpectrumSet
            A new SpectrumSet, containing the smoothed version of the
            current spectra and an updated spectral resolution.
        """
        sigmas = np.asarray(std, dtype=float)
        if np.any(sigmas <= const.float_tol):
            raise ValueError("Invalid smoothing values - Gaussian "
                             "standard deviation must be positive")
        if sigmas.shape != self.spectra.shape:
            raise ValueError("Invalid smoothing shape - must match the shape "
                             "of spectra: {}".format(self.spectra.shape))
        # convolve spectra
        smoothed_spectra = np.zeros((self.num_spectra, self.num_samples))
        for spec_index, spectrum in enumerate(self.spectra):
            mask = self.metaspectra['bad_data'][spec_index, :]
            sigma_iteration = enumerate(zip(self.waves, sigmas[spec_index]))
            for w_index, (output_w, sigma) in sigma_iteration:
                kernel = utl.gaussian(self.waves, output_w, sigma)
                smoothed_value = integ.simps(spectrum[~mask]*kernel[~mask],
                                             self.waves[~mask])
                smoothed_spectra[spec_index, w_index] = smoothed_value
        new_ir = np.sqrt(self.metaspectra["ir"]**2 +
                         (sigmas*const.gaussian_fwhm_over_sigma)**2)
            # TO DO: Implement here some noise estimate in the smoothed
            # spectra to replace the error handling below
        scaled_noise = np.absolute(self.metaspectra["noise"]/self.spectra)
        no_noise = np.all(scaled_noise < const.float_tol)
        if no_noise:
            # can propagate the noise for perfect data
            new_noise = np.zeros(self.metaspectra["noise"].shape)
        else:
            new_noise = np.nan*np.ones(self.num_samples)
            warning.warn("Convolution has no noise propagation - "
                         "noises will be set NaN")
        # crop output
        edge_fwhm = new_ir[:, [0, -1]].max(axis=0) # max over spectra
        edge_sigmas = edge_fwhm/const.gaussian_fwhm_over_sigma
        edge_buffer = crop_factor*edge_sigmas
        valid_interval = self.spec_region + edge_buffer*np.array([1, -1])
        valid = utl.in_linear_interval(self.waves, valid_interval)
        # record smoothing
        extened_comments = self.comments.copy()
        extened_comments["smoothing"] = ("Spectra have been Gaussian-"
                                         "smoothed and cropped to "
                                         "remove edge-effects")
        return SpectrumSet(spectra=smoothed_spectra[:, valid],
                           bad_data=self.metaspectra['bad_data'][:, valid],
                           noise=new_noise[:, valid], ir=new_ir[:, valid],
                           spectra_ids=self.ids, spectra_unit=self.spec_unit,
                           wavelengths=self.waves[valid], name=self.name,
                           wavelength_unit=self.wave_unit,
                           comments=extened_comments)

    def crop(self, region_to_keep):
        """
        Return a new SpectrumSet cropped to include data only in
        the passed wavelength interval.
        """
        region_to_keep = np.asarray(region_to_keep, dtype=float)
        to_keep = utl.in_linear_interval(self.waves, region_to_keep)
        updated_comments = self.comments.copy()
        updated_comments["cropped"] = ("Spectra have been cropped "
                                       "from original wavelength extent")
        return SpectrumSet(spectra=self.spectra[:, to_keep],
                           bad_data=self.metaspectra['bad_data'][:, to_keep],
                           noise=self.metaspectra['noise'][:, to_keep],
                           ir=self.metaspectra['ir'][:, to_keep],
                           spectra_ids=self.ids, spectra_unit=self.spec_unit,
                           wavelengths=self.waves[to_keep], name=self.name,
                           wavelength_unit=self.wave_unit,
                           comments=updated_comments)


    def to_fits_hdulist(self):
        """
        Convert all data to an astropy HDUList object, which can be
        directly written to a .fits file.

        The each extension will hold of the SpectrumSet's data arrays,
        in the order: spectra, noise, wavelengths, bad_data, ir, ids.
        The spectrum comments will be used as .fits header comments.
        """
        baseheader = fits.Header()
        baseheader.append(("dataset", self.name))
        baseheader.append(("specunit", str(self.spec_unit)))
        baseheader.append(("waveunit", str(self.wave_unit)))
        baseheader.append(("primary", "spectra"))
        baseheader.add_comment("spectral resolution given in "
                               "wavelength units, Gaussian FWHM")
        for k, v in self.comments.iteritems():
            baseheader.add_comment("{}: {}".format(k, v))
        hdu_spectra = fits.PrimaryHDU(data=self.spectra, header=baseheader)
        hdu_waves = fits.ImageHDU(data=self.waves,
                                  header=baseheader, name="waves")
        hdu_ids = fits.ImageHDU(data=self.ids, header=baseheader, name="ids")
        hdu_noise = fits.ImageHDU(data=self.metaspectra["noise"],
                                  header=baseheader, name="noise")
        hdu_ir = fits.ImageHDU(data=self.metaspectra["ir"],
                               header=baseheader, name="ir")
        integer_bad_data = self.metaspectra["bad_data"].astype(int)
        hdu_bad_data = fits.ImageHDU(data=integer_bad_data,
                                     header=baseheader, name="bad_data")
        hdulist = fits.HDUList(hdus=[hdu_spectra, hdu_noise, hdu_waves,
                                     hdu_bad_data, hdu_ir, hdu_ids])
        return hdulist

    def write_to_fits(self, path):
        """
        Write all data to a .fits file at the passed location.

        This file will have an extension for each of the SpectrumSet's
        data arrays, in the order: spectra, noise, wavelengths,
        bad_data, ir, ids. Comments are used as .fits header comments.
        """
        hdulist = self.to_fits_hdulist()
        hdulist.writeto(path, clobber=True)


def read_datacube(path, name=None):
    """
    Read a .fits datacube into a SpectrumSet object.

    The format of the .fits is assumed to be that of the MASSIVE
    convention Mitchell datacubes: five extensions, giving the
    spectra, noise, waves, bad_data mask, spectral resolution, and
    id numbers. If a sixth extension is present, likely containing
    the (Ra, Dec) coordinates of each fiber, it will be ignored.
    In each of these arrays, each row holds the data for one fiber and
    the ordering of fibers is assumed to be consistent between all
    extensions. The spectral data are assumed to be in cgs flux per
    angstroms, the wavelength data in angstroms.

    The name of the dataset can be given, otherwise it is taken
    from the file path.
    """
    path = os.path.normpath(path)
    if name is None:
        name = os.path.splitext(os.path.split(path)[-1])[0]
    data, headers = utl.fits_quickread(path)
    [spectra, noise, waves, bad_data, ir, ids] = data  # assumed order
    [spectra_h, noise_h, waves_h, bad_data_h, ir_h, ids_h] = headers
    spec_unit = const.flux_per_angstrom  # Mitchell assumed value
    waves_unit = const.angstrom  # Mitchell assumed value
    # TO DO: remove overwrite in comment concatenation
    comments = {}
    comments.update({k:str(v) for k, v in waves_h.iteritems()})
    comments.update({k:str(v) for k, v in spectra_h.iteritems()})
    return SpectrumSet(spectra=spectra, bad_data=bad_data.astype(bool),
                       noise=noise, ir=ir, spectra_ids=ids,
                       wavelengths=waves, spectra_unit=spec_unit,
                       wavelength_unit=waves_unit, comments=comments,
                       name=name)

