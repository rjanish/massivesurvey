"""
This module handles the storage and common manipulations of spectra
and collections of spectra.
"""


import numpy as np
import astropy.units as units
import scipy.integrate as integrate

import utilities as utl


class SpectrumSet(object):
    """
    This class holds a set of spectra along with their metadata.

    The metadata are the focus here - this class mainly serves to
    enforce that all of the information needed to interpret spectra is
    explicitly recorded along with the spectral data. The included
    methods for I/O and manipulation of spectra are meant to enforce
    the preservation and automatic updating of the metadata as needed.
    """
    def __init__(self, spectra, wavelengths, noise, ir, spectra_unit,
                 wavelength_unit, comments={}, float_tol=10**(-10)):
        """
        Mandatory arguments here force explicit recording of metadata.
        When this function returns, the object will hold all of the
        spectra metadata relevant for kinematic modeling.

        Args:
        spectra - 1d or 2d arraylike
            The actual spectrum values. Each spectrum occupies a row,
            and are assumed to be identical in sampling and units. A
            single 1d array may be passed, which is interpreted as a
            set containing a single spectrum and is converted as such.
        wavelengths - 1d arraylike
            Wavelength sample values, the same for each spectrum.
        noise - arraylike, matches the shape of spectrum
            Estimate of noise level in spectrum, assumed to be the
            half-width of a 1-sigma confidence interval. The units are
            assumed to match those of spectrum.
        ir - arraylike, matches the shape of spectrum
            The spectral resolution of the instrument that recorded
            the spectra. This is assumed to be given as a Gaussian
            FWHM, with the units matching wavelengths.
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
        self.spectra = np.asarray(spectra, dtype=float)
        if self.spectra.ndim not in [1, 2]:
            raise ValueError("Invalid spectra shape: {}. Must have 1 or 2 "
                             "dimensions.".format(self.spectra.shape))
        self.spectra = np.atleast_2d(self.spectra)  # spectrum shape now 2D
        self.num_spectra, self.num_samples = self.spectra.shape
        # check waves format
        self.waves = np.asarray(wavelengths, dtype=float)
        if ((self.waves.ndim != 1) or
            (self.waves.size != self.num_samples)):
            raise ValueError("Invalid wavelength shape: {}. Must be "
                             "1D and match the size of spectra: {}."
                             "".format(self.waves.shape, self.num_samples))
        self.spec_region = np.array([self.waves.min(), self.waves.max()])
        # check metaspectra format
        # 'metaspectra' are those metadata that have a spectra-like form
        metaspectra_inputs = [noise, ir]
        metaspectra_names = ["noise", "ir"]
        conversion = lambda a: np.atleast_2d(np.asarray(a, dtype=float))
        metaspectra_data = map(conversion, metaspectra_inputs)
            # metaspectra are now float-valued and have dimension >= 2
        self.metaspectra = dict(zip(metaspectra_names, metaspectra_data))
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

    def is_linearly_sampled(self):
        """ Check if wavelengths are linear spaced. Boolean output. """
        delta = self.waves[1:] - self.waves[:-1]
        residual = np.absolute(delta - delta[0]).max()
        return residual < self.tol

    def is_log_sampled(self):
        """ Check if wavelengths are log spaced. Boolean output. """
        log_waves = np.log(self.wavelengths)
        delta = log_waves[1:] - log_waves[:-1]
        residual = np.absolute(delta - delta[0]).max()
        return residual < self.tol

    def compute_flux(self, region=None):
        """
        Compute the flux of spectrum over the given region.

        The flux is computed as the integral of the spectrum over
        wavelength. This is done using whatever units are given at
        the time of class construction, so the output of this routine
        is not necessarily expressed in units of flux. The integration
        is done using Simpson's quadrature.

        Args:
        region - 1D, 2-element arraylike; default = full data range
            The wavelength interval over which to compute the flux,
            expressed as an array [lamba_start, lamba_end]. This
            interval must be contained in the data's spectral range.

        Return:
        flux - float
            The wavelength-integrated flux of spectrum.
        flux_unit - astropy unit
            The unit in which flux is given; this will be
            spectrum_unit*wavelength_unit.
        """
        if region is None:
            region = self.spec_region
        elif not utl.interval_contains_interval(self.spec_region, region):
            raise ValueError("Invalid region: {}. Region must be "
                             "contained in data spectral range: {}."
                             "".format(region, self.spec_region))
        valid = utl.in_interval(self.waves, region)
        flux = integrate.simps(self.spectra[valid], self.waves[valid])
        flux_unit = self.spec_unit*self.wave_unit
        return flux, flux_unit
