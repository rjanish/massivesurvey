"""
This module handles storage and common manipulations for libraries
of stellar template spectra.
"""


import os

import utilities as utl
import massivepy.spectrum as spec
import massivepy.constants as const


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


def read_templatelibrary(dirname):
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
    
    spectra_dir = os.path.join()



    return


class TemplateLibrary(spec.SpectrumSet):
    """
    This class holds a stellar template library, and provides methods
    for manipulations commonly needed on such libraries.
    """
    def __init__(self, catalouge=None, **kwargs):
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
        index_array = self.catalog.values
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
        res_matches = np.max(np.absolute(self.metaspectra["ir"] -
                                         target_resolution))
        if res_matches:
            warnings.warn("Templates already have the target spectral "
                          "resolution, skipping convolution", RuntimeWarning)
            return
        increase_res = np.any(self.metaspectra["ir"] > target_resolution)
        if included_res:
            raise ValueError("Invalid target resolution - must be "
                             "greater than the current resolution")
        fwhm_to_add = np.sqrt(target_fwhm**2 - self.metaspectra["ir"]**2)
        sigma_to_add = fwhm_to_add/const.gaussian_fwhm_over_sigma
        self.specset = self.specset.gaussian_convolve(sigma_to_add)
        return