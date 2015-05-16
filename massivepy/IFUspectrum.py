"""
This module handles storage and computations with IFU spectral fields,
i.e. collections of spectra each associated with a spacial region.
"""


import astropy.units as units

import massivepy.spectrum as spec


class IFUspectrum(spec.spectrum):
    """
    This class holds a set of spectra and associated spacial regions.

    The data stored here represent a primitive IFU observation, i.e.
    the spectra are all assumed to have the same spacial footprint
    and to not be the result of binning multiple spectra from
    non-coincident regions.
    """
    def __init__(self, spectrumset, coords, coords_unit, footprint):
        """
        Args:
        spectrumset - SpectrumSet object
            The spectral data, as a SpectrumSet object
        coords - (Nx2) arraylike
            The Cartesian coordinates of the center of each spectrum's
            spacial footprint. The ordering along the 0-axis must
            match that of spectrumset: the (n+1)th spectrum is
            spectrumset.spectra[n, :] and has center coordinates[n, :].
        coords_unit - astropy unit-like
            The unit in which the coordinate values are given.
        footprint - func
            This is function specifies the footprint of the spectra.
            It should accept a pair of central Cartesian coordinates
            as a 1d arraylike, and return a footprint shape as a
            matplotlib Patch centered on the passed coordinates.
        """
        self.spectrumset = spectrumset
        self.coords = np.asarray(coords)
        required_coord_shape = (self.spectrumset.num_spectra, 2)
        if self.coords.shape != required_coord_shape:
            msg = ("Invalid coords shape {}, needs {}"
                   "".foramt(self.coords.shape, required_coord_shape))
            raise ValueError(msg)
        self.coords_unit = units.Unit(coords_unit)
        self.footprint = footprint