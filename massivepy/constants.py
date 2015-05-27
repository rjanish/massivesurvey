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

# MASSIVE conventions
selected_twelve = [14, 227, 303, 311, 547, 561, 586, 591, 052, 755, 844, 871]
    # MILES numbers of the twelve templates chosen by Jenny in November 2014.
    # These were chosen in order to have a defined set of a handful of mostly
    # G and K, with a few M, stars spanning the available MILES range of
    # metallicity. Otherwise, they were chosen by-hand and mostly randomly.
float_tol = 10**(-10)

# math
gaussian_fwhm_over_sigma = 2*np.sqrt(2*np.log(2))

# directory structure
path_to_datamap = "etc/data_locations.txt"