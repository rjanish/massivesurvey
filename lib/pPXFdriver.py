"""
This is a wrapper for ppxf, used to facilitate kinematic fitting.
"""


class pPXFDriver(object):
    """
    This class is a driver for sets of pPXF fits, aiding in particular
    the preparation of spectra and the recording of results.

    Before allowing a pPXF fit to occur, this class will ensure that
    both the observed spectra and template spectra are log-sampled in
    wavelength and have identical spectral resolutions.

    All of the data and settings input to the fit are stored as class
    attributes. Intermediate computations and fit results are
    similarly stored. A pickle of the entire class will serve as an
    exact record of the fit, and I/O methods are provided to simplify
    recording of individual results.
    """
    def __init__(spectra):
        """
        Args:
        spectra - SpectraSet object
            The spectra which will be fit by pPXF. Must be sampled
            with logarithmic spacing and have units of flux/velocity.
        """
        pass