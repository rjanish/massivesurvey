"""
This module handles storage and computations with IFU spectral fields.
"""


import re
import os
import functools
import time

import numpy as np
import shapely.geometry as geo
import astropy.units as units
import astropy.io.fits as fits

import utilities as utl
import massivepy.spectrum as spec
import massivepy.binning as binning
import massivepy.constants as const
import massivepy.io as mpio
import massivepy.spectralresolution as res


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
                   "".format(self.coords.shape, required_coord_shape))
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

    def crop(self, region_to_keep):
        """
        Modify the spectrumset object by cropping to include data
        only in the passed wavelength interval.
        See SpectrumSet.crop for details
        """
        self.spectrumset = self.spectrumset.crop(region_to_keep)
        return

    def align_dithers(self, vlist, mode='interpolate'):
        """
        Modify the spectrumset object by aligning the radial velocities
        of each dither. Accepts a list of radial velocities in km/s and
        shifts the spectra to match these velocities. Will match all other
        dithers to the one with the median velocity.
        See SpectrumSet.shift_subset for details.
        """
        vlist = np.atleast_1d(vlist)
        ndithers = len(vlist)
        if ndithers == 1:
            print "Only one dither, nothing to align."
            return
        elif ndithers%2 == 0:
            vfid = np.nanmedian(vlist[:-1])
        else:
            vfid = np.nanmedian(vlist)
        for i,v in enumerate(vlist):
            if v==vfid:
                continue
            if np.isnan(v):
                continue
            vshift = v - vfid
            wshift = np.sqrt((1+vshift/3.0e5)/(1-vshift/3.0e5))
            ishift_exact = -np.log(wshift)/self.spectrumset.get_logscale()
            # would like to make this round more intelligently
            # e.g. if D1 rounds 1.45 pixels to 1, D2 should round 1.55 pixels
            #  to 1 pixel instead of to 2, so the overall spread is minimized
            ishift = int(np.round(ishift_exact))
            ditherids = 245*i + np.arange(245)
            self.spectrumset.shift_subset(ditherids,wshift,ishift,mode=mode)
        return

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

        See the corresponding function for SpectrumSet. This list
        made by this function will have an addition extension
        giving the spectra coordinates.
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
    THIS FUNCTION HAS BEEN DEPRECATED, use read_raw_datacube instead,
    since the .fits file this is designed for (output of process_mitchell)
    is no longer being created.

    Read a .fits datacube into an IFUspectrum object.

    The format of the .fits is assumed to be that of the MASSIVE
    convention Mitchell datacubes: six extension, giving the spectra,
    noise, waves, bad_data mask, spectral resolution, id numbers,
    and (Ra, Dec) coordinates of each fiber. In each of these arrays,
    each row holds the data for one fiber and the ordering of fibers
    is assumed to be consistent between all extensions. The spectral
    data are assumed to be in cgs flux per angstroms, the
    wavelength data in angstroms, and coords in Ra, Dec degrees.

    The name of the dataset can be given, otherwise it is taken
    from the file path.
    """
    path = os.path.normpath(path)
    if name is None:
        name = os.path.splitext(os.path.split(path)[-1])[0]
    data, headers = utl.fits_quickread(path)
    [spectra, noise, waves,
     bad_data, ir, ids, coords] = data  # assumed order
    [spectra_h, noise_h, waves_h,
     bad_data_h, ir_h, ids_h, coords_h] = headers  # assumed order
    coords_unit = const.degree  # Mitchell assumed values
    linear_scale = const.mitchell_fiber_radius.value
    footprint = lambda center: geo.Point(center).buffer(linear_scale)
    spec_unit = const.flux_per_angstrom  # Mitchell assumed values
    waves_unit = const.angstrom  # Mitchell assumed values
    # TO DO: remove overwrite in comment concatenation
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
    comments = {}
    comments['description'] = ('dimensionless distance in plane through '
                               'galaxy center and perpendicular to line '
                               'of sight, in arcseconds, with origin at '
                               'galaxy center, and with physical '
                               'distance given by (coords/3600)*(pi/180)*'
                               'R where R is distance along line of sight.')
    comments['galaxy center RA'] = center[0]
    comments['galaxy center DEC'] = center[1]
    comments['galaxy center units'] = 'degrees'
    comments['coord units'] = 'arcsec'
    comments['x-direction'] = 'East'
    comments['y-direction'] = 'North'
    return new_coords, comments

def read_raw_datacube(cube_path, gal_info , gal_name, ir_path=None,
                      return_arcs=False):
    """
    This is intended to replace read_mitchell_rawdatacube, since we no longer
    want to save a copy of all of this fiber stuff.

    Comments are populated with important metadata from the fits file headers,
    to be passed on to descendent files. See the following for details:
      spectrum.SpectrumSet.to_fits_hdulist()
      spectralresolution.save_specres()
      s2_bin_mitchell, for bininfo.txt file
    """
    # record all important metadata in the comments
    comments = {}
    coord_comments = {}
    comments['rawfile'] = os.path.basename(cube_path)
    # read and parse ifu file
    data, headers = utl.fits_quickread(cube_path)
    if gal_name=='NGC3842':  # hard code an ugly exception for 3842
        print '\n==================\nUSING DONATED ARCS\n==================\n'
        arcdonor = ('/Users/melanieveale/Box Sync/MASSIVE/Reduced-Data'
                    '/NGC4874/RnovfibNGC4874ALL_log.fits')
        donordata, donorheaders = utl.fits_quickread(arcdonor)
        donorarcs = np.vstack(3*[donordata[4][:246,:]]) # from 2 dithers to 3
        donorinstwaves = donordata[2][0]*(1 + donorheaders[2]['z'])
        recipientinstwaves = data[2][0]*(1 + headers[2]['z'])
        rarcs = np.zeros(data[0].shape) # different wavelength samples, ugh
        for i,eachfiber in enumerate(rarcs):
            rarcs[i] = np.interp(recipientinstwaves,donorinstwaves,donorarcs[i])
        data = data + (rarcs,) # pretend they were there all along!
        headers = headers + (donorheaders[4],) # and the header too
    elif gal_name=='NGC2258':
        print '\n==================\nCENTERING BY HAND\n==================\n'
        data[3][:,0] += 0.008
        data[3][:,1] += 0.0012
    try:
        # wavelengths of arc spectra are specifically included
        spectra, noise, all_waves, coords, arcs, all_inst_waves = data
        spectra_h, noise_h, waves_h, coords_h, arcs_h, inst_waves_h = headers
        gal_waves = all_waves[0, :]  # assume uniform samples; gal rest frame
        inst_waves = all_inst_waves[0, :]  # instrument rest frame, not used
        redshift = waves_h['z']  # assumed redshift of galaxy
    except ValueError:
        # wavelength of arc spectra not included - compute by shifting
        # the spectra wavelength back into the instrument rest frame
        spectra, noise, all_waves, coords, arcs = data
        spectra_h, noise_h, waves_h, coords_h, arcs_h = headers
        gal_waves = all_waves[0, :]  # assume uniform samples; gal rest frame
        redshift = waves_h['z']  # assumed redshift of galaxy
        inst_waves = gal_waves*(1 + redshift)  # instrument rest frame, not used
    comments['rawdate'] = time.ctime(os.path.getmtime(cube_path))
    comments['redshift'] = redshift
    nfibers, npixels = spectra.shape
    wavelengths = gal_waves
    comments['frame'] = 'galaxy rest frame'
    bad_data = (spectra==-6.66e19)|np.isnan(spectra)
    ir = np.nan*np.ones(spectra.shape)
    # we assume units won't change, but check anyway
    if not spectra_h['bunit'] == 'ergs/s/cm^2/A':
        raise Exception("Unexpected flux units in datacube!")
    if not waves_h['bunit'] == 'Angstrom':
        raise Exception("Unexpected wavelength units in datacube!")
    if not coords_h['bunit'] == 'Deg':
        raise Exception("Unexpected coordinate units in datacube!")
    # convert coordinates and stuff
    coord_comments['galaxy pa'] = gal_info['pa']
    coord_comments['galaxy pa units'] = 'degrees E of N'
    gal_center = (gal_info['ra'],gal_info['dec'])
    cart_coords, cart_comments = center_coordinates(coords, gal_center)
    coord_comments.update(cart_comments)
    linear_scale = const.mitchell_fiber_radius.value # assuming units match!
    footprint = lambda center: geo.Point(center).buffer(linear_scale)
    coord_comments['fiber shape'] = 'circle'
    ifuset = IFUspectrum(spectra=spectra, # kwargs required by SpectrumSet
                         bad_data=bad_data,
                         noise=noise,
                         ir=ir,
                         spectra_ids=np.arange(nfibers),
                         wavelengths=wavelengths,
                         spectra_unit=const.flux_per_angstrom,
                         wavelength_unit=const.angstrom,
                         comments=comments,
                         name=gal_name,
                         coords=cart_coords, # kwargs specific to IFUspectrum
                         coords_unit=const.arcsec,
                         coord_comments=coord_comments,
                         footprint=footprint,
                         linear_scale=linear_scale)
    # remove bad fibers and add in real ir if available
    bad_fibers = np.where(np.all(bad_data,axis=1))[0]
    if len(bad_fibers)>0:
        print 'Entire fiber is bad for the following, skipping them entirely!'
        print bad_fibers
    good_fibers = np.where(~np.all(bad_data,axis=1))[0]
    if gal_name=='NGC4874': # toss first fiber of 4874, is a special case
        print 'Also tossing fiber 0 because of weird coordinate thing'
        good_fibers = good_fibers[1:]
    elif gal_name=='NGC3842':
        good_fibers = list(good_fibers)
        good_fibers.remove(125)
        good_fibers.remove(371)
        good_fibers.remove(617)
        #good_fibers = np.array(good_fibers)
    ifuset = ifuset.get_subset(good_fibers)
    arcs = arcs[good_fibers,:]
    if not ir_path is None:
        ir_samples = res.read_specres(ir_path)
        # assuming ir_samples is in instrument rest frame
        # res.specres_for_galaxy will shift it to galaxy frame to match spectra
        ir = res.specres_for_galaxy(ir_samples['fitcenter'],ir_samples['fwhm'],
                                    wavelengths, redshift)
        ircomments = {'irfile': os.path.basename(ir_path),
                      'irdate': time.ctime(os.path.getctime(ir_path))}
        ifuset.spectrumset.metaspectra['ir'] = ir
        ifuset.spectrumset.comments.update(ircomments)
    if return_arcs:
        return ifuset, arcs
    else:
        return ifuset
