"""
This module handles the storage and common manipulations of spectra
and collections of spectra.
"""


import numpy as np


class SpectrumSet(object):
    """
    This class holds a set of spectra along with their metadata.

    The metadata are the focus here - this class mainly serves to
    enforce that all of the information needed to interpret spectra is
    explicitly recorded along with the spectral data. The included
    methods for I/O and manipulation of spectra are meant to enforce
    the preservation and automatic updating of the metadata as needed.
    """
    def __init__(self, spectra, wavelengths, noise, ir,
                 comments={}, float_tol = 10**(-10)):
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
        comments - dict, default is empty
            This is a dictionary to store comments about the spectra.
            The keys are treated as strings, otherwise there are no
            formatting restrictions. These comments will be shuffled
            along with I/O operations in file headers.
        float_tol - float, default = 10^(-10)
            The relative tolerance used for floating-point comparison.
        """
        # check spectra format
        self.spectra = np.asarray(spectra)
        if self.spectra.ndim == 1:
            self.spectra = np.expand_dims(spectra, 0)
        elif self.spectra.ndim != 2:
            raise ValueError("Invalid spectra shape: {}. Must have 1 or 2 "
                             "dimensions.".format(self.spectra.shape))
        self.num_spectra, self.num_samples = self.spectra.shape
        # check waves format
        self.waves = np.asarray(wavelengths)
        if ((self.waves.ndim != 1) or
            (self.waves.size != self.num_samples)):
            raise ValueError("Invalid wavelength shape: {}. Must be "
                             "1D and match the size of spectra: {}."
                             "".format(self.waves.shape, self.num_samples))
        # check metaspectra format
        # 'metaspectra' are those metadata that have a spectra-like form
        metaspectra_data = map(np.asarray, [noise, ir])
        metaspectra_names = ["noise", "ir"]
        self.metaspectra = dict(zip(metaspectra_names, metaspectra_data))
        for name, data in self.metaspectra.iteritems():
            if data.ndim == 1:
                self.metaspectra[name] = np.expand_dims(data, 0)
            if data.shape != self.spectra.shape:
                error_msg = ("Invalid {} shape: {}. "
                             "Must match the shape of spectra: {}."
                             "".format(name, data.shape, self.spectra.shape))
                raise ValueError(error_msg)
        # remaining arg checks
        self.comments = {str(k):v for k, v in comments.iteritems()}
        self.tol = float(float_tol)

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