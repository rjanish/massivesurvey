"""
Miscellaneous constants and unit definitions of general use.
"""


import astropy.units as units

arcsec = units.arcsec
cgs_flux = units.erg/(units.second*(units.cm**2))
angstrom = units.angstrom
flux_per_angstrom = cgs_flux/units.angstrom