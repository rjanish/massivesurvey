"""
This module handles storage and common manipulations for libraries
of stellar template spectra.
"""


import massive.spectrum as spec


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