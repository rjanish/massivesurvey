"""
Miscellaneous constants and unit definitions of general use.
"""


import numpy as np
import astropy.units as units


# units
arcsec = units.arcsec
cgs_flux = units.erg/(units.second*(units.cm**2))
angstrom = units.angstrom
flux_per_angstrom = cgs_flux/units.angstrom

# instrument constants
mitchell_fiber_radius = 2.08 * arcsec
mitchell_arc_centers = np.array([4046.5469,
                                 4077.8403,
                                 4358.3262,
                                 4678.1558,
                                 4799.9038,
                                 4916.0962,
                                 5085.8110,
                                 5460.7397,
                                 5769.5972])
mitchell_nominal_spec_resolution = 4.6  # Angstroms, Gaussian FWHM

# MASSIVE conventions
selected_twelve = [14, 227, 303, 311, 547, 561, 586, 591, 052, 755, 844, 871]
    # MILES numbers of the twelve templates chosen by Jenny in November 2014.
    # These were chosen in order to have a defined set of a handful of mostly
    # G and K, with a few M, stars spanning the available MILES range of
    # metallicity. Otherwise, they were chosen by-hand and mostly randomly.
float_tol = 10**(-10)
re_ngc = r"(?:NGC|ngc|N)(?P<num>\d{4})"

# math
gaussian_fwhm_over_sigma = 2*np.sqrt(2*np.log(2))

# directory structure
path_to_datamap = "etc/datamap.txt"