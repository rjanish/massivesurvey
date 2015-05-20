"""
This module handles storage and computations with IFU spectral fields,
i.e. collections of spectra each associated with a spacial region.
"""


import re

import astropy.units as units
import numpy as np

import massivepy.spectrum as spec
import massivepy.binning as binning


class IFUspectrum(object):
    """
    This class holds a set of spectra and associated spacial regions.

    The data stored here represent a primitive IFU observation, i.e.
    the spectra are all assumed to have the same spacial footprint
    and to not be the result of binning multiple spectra from
    non-coincident regions.
    """
    def __init__(self, coords=None, coords_unit=None,
                 footprint=None, linear_scale=None, **kwargs):
        """
        See SpectrumSet. Arguments needed beyond those of SpectrumSet
        are described below.

        Args:
        spectrumset - SpectrumSet object or keyword args
            The spectral data, either as a SpectrumSet object, or by
            passing all of the keyword arguments needed by SpectrumSet
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
            shapely polygon object centered on the passed coordinates.
        linear_scale - float
            -
        """
        if 'spectrumset' in kwargs:
            self.spectrumset = kwargs['spectrumset']
        else:
            self.spectrumset = spec.SpectrumSet(**kwargs)
        self.coords = np.asarray(coords)
        required_coord_shape = (self.spectrumset.num_spectra, 2)
        if self.coords.shape != required_coord_shape:
            msg = ("Invalid coords shape {}, needs {}"
                   "".foramt(self.coords.shape, required_coord_shape))
            raise ValueError(msg)
        self.coords_unit = units.Unit(coords_unit)
        self.footprint = footprint
        self.linear_scale = float(linear_scale)

    def get_subset(self, ids):
        """
        Extract subset of spectral data with the passed spectrum ids.
        The associated spectra coordinates will also be extracted.

        For details, see SpectrumSet.get_subset.
        """
        new_set, index = self.spectrumset.get_subset(ids, get_selector=True)
        new_coords = self.coords[index, :]
        return IFUspectrum(spectrumset=new_set, coords=new_coords,
                           coords_unit=self.coords_unit,
                           footprint=self.footprint)

    def s2n_spacial_binning(self, threshold=None, binning_func=None,
                            combine_func=None):
        """
        """
        binned = binning_func(
            collection=self.spectrumset, coords=self.coords,
            ids=self.spectrumset.ids, linear_scale=self.linear_scale,
            indexing_func=spec.SpectrumSet.get_subset,
            combine_func=combine_func, threshold=threshold,
            score_func=spec.SpectrumSet.compute_mean_s2n)
        return binned