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
        sigma_to_add = fwhm_to_add/(2*np.sqrt(2*np.log(2)))


    # compute convolution weights
    fwhm_to_add = np.sqrt(target_fwhm**2 - current_fwhm**2)
    sigma_to_add = fwhm_to_add/(2*np.sqrt(2*np.log(2)))
    measure = wavelengths[1:] - wavelengths[:-1]
    weights = []
    for central_wavelength, sigma in zip(wavelengths, sigma_to_add):
        trapz_samples = np.array([wavelengths[:-1], wavelengths[1:]])
            # sample points needed for trapezoid integration
        psf = gaussian(trapz_samples, central_wavelength, sigma)
        weights.append(psf*measure*0.5)
    weights = np.swapaxes(weights, 0, 1)
    # apply convolution, log transform, re-sampling, and normalization
    finished_spectra = []
    for counter, spectrum in enumerate(spectra):
        smoothed_spectrum = (weights[0, ...]*spectrum[:-1] +
                             weights[1, ...]*spectrum[1:]).sum(axis=1)
            # trapezoid integration of continuous convolution integral
        interpolator = interp1d(wavelengths, smoothed_spectrum)
        interpolated_spectrum = interpolator(logspaced_waves)
        edge_sigmas = target_fwhm[[0, -1]]/(2*np.sqrt(2*np.log(2)))
        edge_buffer = crop_factor*edge_sigmas
        valid = ((wavelengths.min() + edge_buffer[0] < logspaced_waves) &
                 (logspaced_waves < wavelengths.max() - edge_buffer[1]))
        interpolated_spectrum = interpolated_spectrum[valid]
        logspaced_waves = logspaced_waves[valid]
            # crop to eliminate edge effects

        norm = np.median(interpolated_spectrum)
        finished_spectra.append(interpolated_spectrum/norm)
