"""
This module handles storage and computations with IFU spectral fields,
i.e. collections of spectra each associated with a spacial region.
"""


import re
import os
import functools

import numpy as np
import shapely.geometry as geo
import astropy.units as units
import astropy.io.fits as fits

import utilities as utl
import massivepy.spectrum as spec
import massivepy.binning as binning
import massivepy.constants as const


class IFUspectrum(object):
    """
    This class holds a set of spectra and associated spacial regions.

    The data stored here represent a primitive IFU observation, i.e.
    the spectra are all assumed to have the same spacial footprint
    and to not be the result of binning multiple spectra from
    non-coincident regions.
    """
    def __init__(self, coords=None, coords_unit=None, coord_comments={},
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
            A characteristic linear dimension of the spacial footprint
        coord_comments - dict, default empty
            Comments about the coordinate system
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
        self.coord_comments = coord_comments

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
                           footprint=self.footprint,
                           linear_scale=self.linear_scale,
                           coord_comments=self.coord_comments)

    def s2n_fluxweighted_binning(self, threshold=None, get_bins=None):
        """
        Construct a spacial partition of the spectra into bins,
        according to a S/N threshold.

        Args:
        threshold - float
            The s2n threshold above which bins are valid
        get_bins - func
            A binning algorithm function, which must accept an a
            collection of data and methods for indexing, combining,
            and scoring those collections, and then produce a set of
            bins. See the module 'binning' for detailed examples.
        """
        delta_lambda = (self.spectrumset.spec_region[1] -
                        self.spectrumset.spec_region[0])
        combine = functools.partial(spec.SpectrumSet.collapse, id=0,
                                    weight_func=spec.SpectrumSet.compute_flux,
                                    norm_func=spec.SpectrumSet.compute_flux,
                                    norm_value=delta_lambda)
            # this is flux-normed, flux-weighted mean (i.e, coaddition), with
            # fluxes set so that the numerical spectra values are near 1
        partition = get_bins(collection=self.spectrumset,
                             coords=self.coords, ids=self.spectrumset.ids,
                             linear_scale=self.linear_scale,
                             indexing_func=spec.SpectrumSet.get_subset,
                             combine_func=combine, threshold=threshold,
                             score_func=spec.SpectrumSet.compute_mean_s2n)
        return partition

    def to_fits_hdulist(self):
        """
        Convert all data to an astropy HDUList object, which can be
        directly written to a .fits file.
        """
        hdulist = self.spectrumset.to_fits_hdulist()
        coords_header = fits.Header()
        coords_header.append(("coordunit", str(self.coords_unit)))
        for k, v in self.coord_comments.iteritems():
            # add all coord_comments as header comments
            coords_header.add_comment("{}: {}".format(k, v))
        hdu_coords = fits.ImageHDU(data=self.coords,
                                   header=coords_header, name="coords")
        hdulist.append(hdu_coords)
        return hdulist

    def write_to_fits(self, path):
        """
        Write all data to a .fits file at the passed location.

        See the corresponding function for SpectrumSet. The file
        written by this function will have an addition extension
        giving the spectra coordinates.
        """
        hdulist = self.to_fits_hdulist()
        hdulist.writeto(path, clobber=True)


def read_mitchell_datacube(path, name=None):
    """
    """
    path = os.path.normpath(path)
    if name is None:
        name = os.path.splitext(os.path.split(path)[-1])[0]
    data, headers = utl.fits_quickread(path)
    [spectra, noise, waves,
     bad_data, ir, ids, coords] = data  # assumed order
    [spectra_h, noise_h, waves_h,
     bad_data_h, ir_h, ids_h, coords_h] = headers  # assumed order
    coords_unit = const.angstrom  # Mitchell assumed values
    linear_scale = const.mitchell_fiber_radius.value
    footprint = lambda center: geo.Point(center).buffer(linear_scale)
    spec_unit = const.flux_per_angstrom
    waves_unit = const.angstrom
    # TO DO: remove overwrite in comment concat
    comments = {}
    comments.update({k:str(v) for k, v in waves_h.iteritems()})
    comments.update({k:str(v) for k, v in spectra_h.iteritems()})
    return IFUspectrum(coords=coords, coords_unit=coords_unit,
                       footprint=footprint, linear_scale=linear_scale,
                       coord_comments=dict(coords_h), spectra=spectra,
                       bad_data=bad_data.astype(bool), noise=noise,
                       ir=ir, spectra_ids=ids, wavelengths=waves,
                       spectra_unit=spec_unit, wavelength_unit=waves_unit,
                       comments=comments, name=name)


def center_coordinates(coords, center):
    """
    Given a set of sky coordinates and a center point, transform the
    coordinates into a Cartesian system centered on the passed center.
    The new coordinates are the projection of the old onto the plane
    passing through center and perpendicular to the line of sight.

    Args:
    coords - 2d arraylike, shape (N, 2)
        An array of N sky coordinates (RA, Dec), in degrees, to be
        transformed to a Cartesian system.
    center - 1d arraylike, shape (2,)
        The origin of the new coordinate system, in (Ra, Dec) degrees

    Returns: new_coords
    new_coords - 2d arraylike, same shape as passed coords
        The transformed coordinates. They are dimensionless, given the
        distance in the plane (see above) measured in units of the
        distance along the light-of-sight to the plane, and expressed
        in arcseconds. The first coordinate is the scaled-distance
        along East and the second North. [Physical distances in the
        plane are given by the rescaling (new_coords/3600)*(pi/180)*R,
        with R the distance along the light-of-sight to the plane.]
    """
    coords = np.asarray(coords, dtype=float)
    num_coords, dim = coords.shape
    if dim != 2:
        raise ValueError("Invalid coordinate shape {}, must "
                         "have shape (N, 2)".format(coords.shape))
    center = np.asarray(center, dtype=float)
    if center.shape != (2,):
        raise ValueError("Invalid center coordinate shape {}, must "
                         "have shape (2,)".format(center.shape))
    new_coords = np.zeros(coords.shape)
    new_coords[:, 0] = coords[:, 0] - center[0]  # degrees
    new_coords[:, 1] = coords[:, 1] - center[1]  # degrees
    new_coords[:, 0] = new_coords[:, 0]*np.cos(np.deg2rad(center[1]))
    new_coords *= 60*60  # to arcseconds
    return new_coords