"""
This module handles the storage and common manipulations of spectra
and collections of spectra.
"""


import numpy as np
import astropy.units as units
import scipy.integrate as integ
import scipy.interpolate as inter

import utilities as utl


flux_cgs = units.erg/(units.second*(units.cm**2))
fluxdens = flux_cgs/units.angstrom


class SpectrumSet(object):
    """
    This class holds a set of spectra along with their metadata.

    The metadata are the focus here - this class mainly serves to
    enforce that all of the information needed to interpret spectra is
    explicitly recorded along with the spectral data. The included
    methods for I/O and manipulation of spectra are meant to enforce
    the preservation and automatic updating of the metadata as needed.
    """
    def __init__(self, spectra, bad_data, noise, ir, spectra_ids,
                 wavelengths, spectra_unit, wavelength_unit,
                 comments={}, float_tol=10**(-10)):
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
        float_tol - float, default = 10^(-10)
            The relative tolerance used for floating-point comparison.
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
            (self.ids.size != self.num_samples)):
            raise ValueError("Invalid spectra ids shape: {}. Must be "
                             "1D and match the number of spectra: {}."
                             "".format(self.ids.shape, self.num_spectra))
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
        self.metaspectra = {name:conversions[name](data)
                            for name, data in metaspectra_inputs.iteritems()}
            # metaspectra now float or bool valued and have dimension >= 2
        for name, data in self.metaspectra.iteritems():
            if data.shape != self.spectra.shape:
                error_msg = ("Invalid {} shape: {}. "
                             "Must match the shape of spectra: {}."
                             "".format(name, data.shape, self.spectra.shape))
                raise ValueError(error_msg)
        # remaining arg checks
        self.comments = {str(k):v for k, v in comments.iteritems()}
        self.tol = float(float_tol)
        self.spec_unit = units.Unit(spectra_unit)
        wavelength_unit.to(units.cm)  # check if wavelength_unit is a length
        self.wave_unit = units.Unit(wavelength_unit)

    def __getitem__(self, index):
        """
        Extract a subset of spectral data.

        Returns a new spectrumset object containing a subset of the
        spectra and associated metadata from the original spectrumset.
        Metadata that is by-assumption uniform over all spectra, such
        as the wavelengths, spectral unit, etc., is also preserved.

        Selecting a subset is done by numpy-style indexing into the
        various data arrays, e.g.
        > specset = spectrumset(...)
        > specset[0]
            - returns the first spectrum in specset.spectra
        > specset[1:-1]
            - returns all but the first and last spectra
        > specset[(specset.ids % 2) == 0]
            - returns all spectra with even-number ids

        It is particularly important to keep in mind the difference
        between the index of a spectrum as stored in a spectrumset
        object and the id value assigned to that spectrum:
        > specset = spectrumset(...)
        > numbers = [5, 10, 11, 19]
        > specset[numbers]
            - returns spectra at indices 5, 10, 11, and 19
        > ids_selector = np.in1d(specset.ids, numbers)
        > specset[ids_selector]
            - returns spectra that have id value of 5, 10, 11, or 19
        """
        return SpectrumSet(self.spectra[index, :],
                           self.metaspectra["bad_data"][index, :],
                           self.metaspectra["noise"][index, :],
                           self.metaspectra["ir"][index, :], self.ids[index],
                           self.waves, self.spec_unit, self.wave_unit,
                           comments=self.comments, float_tol=self.tol)

    def is_linear_sampled(self):
        """ Check if wavelengths are linear spaced. Boolean output. """
        delta = self.waves[1:] - self.waves[:-1]
        residual = np.absolute(delta - delta[0]).max()
        return residual < self.tol

    def is_log_sampled(self):
        """ Check if wavelengths are log spaced. Boolean output. """
        log_waves = np.log(self.waves)
        delta = log_waves[1:] - log_waves[:-1]
        residual = np.absolute(delta - delta[0]).max()
        return residual < self.tol

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
        for spec_index in xrange(self.num_spectra):
            spec_func = inter.interp1d(self.waves, self.spectra[spec_index])
            self.spectra[spec_index] = spec_func(new_waves)
            for name, mspec in self.metaspectra.iteritems():
                mspec_func = inter.interp1d(self.waves, mspec[spec_index])
                new_mspec_values = mspec_func(new_waves)
                if name == 'bad_data':
                    new_mspec_values = new_mspec_values.astype(bool)
                        # re-sampled data is valid only if the nearest
                        # bracketing old-sampling values are both valid
                self.metaspectra[name][spec_index] = new_mspec_values
        self.waves = new_waves
        self.num_samples = new_waves.shape[0]
        self.spec_region = new_spec_region
        return

    def log_resample(self, logscale=None):
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
        logscale - float, default=None
            If given, spectra will be re-sampled using the passed
            logscale. If not specified, re-sampling will instead
            preserve the number of data points.

        Return: logscale
        logscale - float
            The logscale of the now re-sampled spectrum
        """
        if self.is_log_sampled():
            raise ValueError("Spectrum is already log-sampled.")
        new_spec_region = self.spec_region*(1 + np.array([1, -1])*self.tol)
            # prevents unintended extrapolation due to roundoff
        log_ends = np.log(new_spec_region)
        if logscale is None:  # preserve sample number
            log_w = np.linspace(log_ends[0], log_ends[1], self.num_samples)
            logscale = log_w[1] - log_w[0]
        else:  # fix log-spacing
            log_w = np.arange(log_ends[0], log_ends[1], logscale)
        new_waves = np.exp(log_w)
        resample(self, new_waves)
        return logscale

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
        new_ends = self.spec_region*(1 + np.array([1, -1])*self.tol)
            # prevents unintended extrapolation due to roundoff
        if step is None:  # preserve sample number
            new_waves = np.linspace(new_ends[0], new_ends[1],
                                    self.num_samples)
            step = new_waves[1] - new_waves[0]
        else:  # fix step size
            new_waves = np.arange(new_ends[0], new_ends[1], step)
        resample(self, new_waves)
        return step

    def compute_flux(self, interval=None, ids=self.ids):
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
        flux_unit - astropy unit
            The unit in which flux is given; this will be
            spectrum_unit*wavelength_unit.
        """
        if interval is None:
            interval = self.spec_region
        ids = np.asarray(ids, dtype=int)
        elif not utl.interval_contains_interval(self.spec_region, interval):
            raise ValueError("Invalid interval: {}. Region must be "
                             "contained in data spectral range: {}."
                             "".format(region, self.spec_region))
        flux_region = utl.in_linear_interval(self.waves, interval)
        fluxes = np.zeros(ids.size)
        for result_index, id in enumerate(ids):
            spec_index = (self.ids == id)
            valid_data = ~self.metaspectra["bad_data"][spec_index]
            to_integrate = flux_region & valid_data
            flux = integ.simps(self.spectra[spec_index, to_integrate],
                               self.waves[to_integrate])
            fluxes[result_index] = flux
        flux_unit = self.spec_unit*self.wave_unit
        return fluxes, flux_unit
