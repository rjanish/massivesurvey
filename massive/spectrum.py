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
    def __init__(self, spectra, wavelengths, comments={}):
        """
        Mandatory arguments here force explicit recording of metadata.
        When this function returns, the object will hold all of the
        spectra metadata relevant for kinematic modeling.

        Args:
        spectra - 2d arraylike
            The actual spectrum values. Each spectrum occupies a row,
            and are assumed to be identical in sampling and units. A
            single 1d array may be passed, which is interpreted as a
            set containing a single spectrum and is converted as such.
        wavelengths - 1d arraylike
            Wavelength sample values, the same for each spectrum.
        comments - dict, default is empty
            This is a dictionary to store comments about the spectra.
            The keys are treated as strings, otherwise there are no
            formatting restrictions. These comments will be shuffled
            along with I/O operations in file headers.
        """
        # check spectra
        self.spectra = np.asarray(spectra)
        if self.spectra.ndim == 1:
            self.spectra = np.expand_dims(spectra, 0)
        elif self.spectra.ndim != 2:
            raise ValueError("Invalid data array shape: {}. Must be either "
                             "1 or 2 dimensions.".format(self.spectra.shape))
        self.num_spectra, self.num_samples = self.spectra.shape
        # check waves
        self.waves = np.asarray(wavelengths)
        if ((self.waves.ndim != 1) or
            (self.waves.size != self.num_samples)):
            raise ValueError("Invalid wavelength array shape: {}. "
                             "Must be 1D and match the size of spectra: {}."
                             "".format(self.waves.shape, self.num_samples))
        # remaining arg checks
        self.comments = {str(k):v for k, v in comments.iteritems()}