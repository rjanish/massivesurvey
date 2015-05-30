"""
This module handles storage and common computation on libraries of
template spectra, both in general and for specific libraries.
"""


import os
import warnings

import numpy as np
import pandas as pd

import utilities as utl
import massivepy.spectrum as spec
import massivepy.constants as const


class TemplateLibrary(object):
    """
    This class holds a stellar template library, and provides methods
    for manipulations commonly needed on such libraries.
    """
    def __init__(self, catalog=None, **kwargs):
        """
        See SpectrumSet. Arguments needed beyond those of SpectrumSet
        are described below.

        Args:
        spectrumset - SpectrumSet object or keyword args
            The spectral data, either as a SpectrumSet object, or by
            passing all of the keyword arguments needed by SpectrumSet
        catalog - pandas dataframe
            A dataframe giving the properties of the spectra in the
            template library. The columns of the dataframe are assumed
            to be the properties, and the index of the dataframe must
            match in order the ids of the accompanying SpectrumSet.
        """
        if 'spectrumset' in kwargs:
            self.spectrumset = kwargs['spectrumset']
        else:
            self.spectrumset = spec.SpectrumSet(**kwargs)
        self.catalog = pd.DataFrame(catalog)
        index_array = self.catalog.index.values
        index_matches = np.all(index_array == self.spectrumset.ids)
        if not index_matches:
            raise ValueError("Invalid catalog index - does "
                             "not match the given spectral ids")
            
    def get_subset(self, ids):
        """
        Extract subset of the library with the passed spectrum ids. A
        new TemplateLibrary will be returned.

        For details, see SpectrumSet.get_subset.
        """
        new_set, index = self.spectrumset.get_subset(ids, get_selector=True)
        new_catalog = self.catalog[index]
        return TemplateLibrary(spectrumset=new_set, catalog=new_catalog)

    def match_resolution(self, target_resolution):
        """
        Convolve the library spectra until their spectra resolution
        matches that passed target_resolution. Does not return -
        modifies the library spectra in place.

        Args:
        target_resolution - 1D arraylike
            The desired spectral resolution, sampled with wavelength
            identically to the library spectra. The units are assumed
            to be Gaussian FWHM in the wavelength units of the library.
        """
        target_fwhm = np.asarray(target_resolution)
        current_resolution = self.spectrumset.metaspectra["ir"]
        delta_var = target_resolution**2 - current_resolution**2
        res_matches = np.max(np.absolute(delta_var)) < const.float_tol
        if res_matches:
            warnings.warn("Templates already have the target spectral "
                          "resolution, skipping convolution", RuntimeWarning)
            return
        decrease_fwhm = np.any(delta_var < 0.0)
        if decrease_fwhm:
            raise ValueError("Invalid target resolution - must be "
                             "greater than the current resolution")
        fwhm_to_add = np.sqrt(delta_var)
        sigma_to_add = fwhm_to_add/const.gaussian_fwhm_over_sigma
        new_specset = self.spectrumset.gaussian_convolve(sigma_to_add)
        return TemplateLibrary(spectrumset=new_specset, catalog=self.catalog)


def miles_number_to_filename(num):
    """
    Convert an id number from the MILES stellar template library into
    the matching default filename (tail only) that holds the spectrum.

    Example:
    > miles_filename_conversion(42)
    'm0042V'
    """
    return "m{:04d}V".format(int(num))


def miles_filename_to_number(filename):
    """
    Convert a default spectra filename (tail only) from the MILES
    stellar template library into the matching MILES id number.

    Example:
    > miles_filename_conversion('m0042V')
    42
    """
    return int(filename[1:-1])


def read_miles_library(dirname):
    """
    Read the template library located at the passed dirname into a new
    TemplateLibrary object.

    This assume a library directory structure:
    dirname/
      spectra/
        - contains one ascii file per spectrum, with wavelengths [A]
          in first column and spectrum [flux(cgs)/A] in second column
      README.txt
        - readable ascii dict file, contains value of the wavelength-
          independent spectral resolution (Gaussian FWHM) [A]
      catalog.txt
        - ascii file giving the parameters of each spectrum, readable
          as a pandas DataFrame
    """
    dirname = os.path.normpath(dirname)
    spectra_dir = os.path.join(dirname, "spectra")
    spectra_paths = utl.re_filesearch(r".*", spectra_dir)[0]
        # output is sorted by filename - i.e., miles id number
    spectra, all_waves, ids = [], [], []
    for path in spectra_paths:
        id = miles_filename_to_number(os.path.basename(path))
        w, s = np.loadtxt(path).T
        spectra.append(s)
        all_waves.append(w)
        ids.append(id)
    spectra = np.asarray(spectra, dtype=float) # all arrays sorted by miles id
    all_waves = np.asarray(all_waves, dtype=float)
    ids = np.asarray(ids, dtype=int)
    wave_frac_spread = all_waves.std(axis=0)/all_waves.mean(axis=0)
    uniform_waves = wave_frac_spread.max() < const.float_tol
    if not uniform_waves:
        return all_waves
        raise ValueError("Spectra must have uniform wavelength sampling")
    waves = all_waves[0]
    readme_path = os.path.join(dirname, "README.txt")
    readme_data = utl.read_dict_file(readme_path)
    fwhm = readme_data["library_fwhm"]
    resolution = np.ones(spectra.shape, dtype=float)*fwhm
        # assume uniform, wavelength-independent template resolution
    catalog_path = os.path.join(dirname, "catalog.txt")
    catalog = pd.read_csv(catalog_path, index_col='miles_id')
        # indexing by miles id will sort DataFrame by miles id
    no_noise = np.zeros(spectra.shape, dtype=float)
    all_good = np.zeros(spectra.shape, dtype=bool)
        # assume perfect template data
    junk, library_name = os.path.split(dirname)
    comments = {}
    comments["original fwhm"] = "{} A".format(fwhm)
    comments["base library"] = "MILES - empirical stellar template library"
    comments["library location"] = dirname
    # return catalog, ids
    return TemplateLibrary(spectra=spectra, bad_data=all_good,
                           noise=no_noise, ir=resolution,
                           spectra_ids=ids, wavelengths=waves,
                           spectra_unit=const.flux_per_angstrom,
                           wavelength_unit=const.angstrom, comments=comments,
                           name=library_name, catalog=catalog)