"""
Miscellaneous constants and unit definitions.
"""


import numpy as np
import astropy.units as units
import astropy.constants as astro_const


# units
arcsec = units.arcsec
degree = units.degree
cgs_flux = units.erg/(units.second*(units.cm**2))
angstrom = units.angstrom
flux_per_angstrom = cgs_flux/units.angstrom
c_kms = astro_const.c.to('km/s').value

# instrumental constants
mitchell_fiber_radius = 2.08 * arcsec
mitchell_arc_centers = np.array([4046.5469,  # Hg-Cd blend
                                 4077.8403,  # Hg
                                 4358.3262,  # Hg
                                 4678.1558,  # Hg-Cd blend -dominated by Hg
                                 4799.9038,  # Cd
                                 5085.8110,  # Cd
                                 5460.7397,  # Hg-Cd blend -dominated by Hg
                                 5769.5972]) # Hg
mitchell_nominal_spec_resolution = 4.6  # Angstroms, Gaussian FWHM
mitchell_crop_region = [3650, 5650] # Angstrom, galaxy rest frame
    # edges can have goofy extrapolated data, this is a stable safe region

# some lines of interest to plot with spectra
# x, y to be used with plt.text; y-offset will usually need normalizing
emission_lines = {'O2': {'wave':3727, 'name':'OII', 'x':3728, 'y':0},
                  'O3a': {'wave':4959, 'name':'OIII(a)', 'x':4960, 'y': 0},
                  'O3b': {'wave':5007, 'name':'OIII(b)', 'x':5008, 'y': 1},
                  'Hb': {'wave':4861, 'name':r'H$\beta$', 'x':4862, 'y':0}}

# MASSIVE conventions
selected_twelve = [14, 227, 303, 311, 547, 561, 586, 591, 052, 755, 844, 871]
    # MILES numbers of the twelve templates chosen by Jenny in November 2014.
    # These were chosen in order to have a defined set of a handful of mostly
    # G and K, with a few M, stars spanning the available MILES range of
    # metallicity. Otherwise, they were chosen by-hand and mostly randomly.
fullMILES_1600fullgalaxy_optimized = [136, 246, 376, 498, 720, 838,
                                      956, 197, 252, 409, 548, 780,
                                      853, 225, 322, 455, 656, 835,  881]
    # DO NOT KEEP HERE - FOR TESTING PURPOSED ONLY
    # Move this to an update-able store of similar list for all galaxies!
    # These are the nonzero templates resulting from a ppxf of the full-
    # galaxy bin of NGC1600 using the full miles-massive template library
fullMILES_5557fullgalaxy_optimized_OLD = [
    71, 107, 204, 409, 472, 614, 792, 931, 969, 93, 149, 219, 427, 498,
    642, 853, 932, 98, 177, 322, 452, 511, 720, 879, 933, 102, 183, 343,
    455, 544, 728, 927, 961]
    # DO NOT KEEP HERE - FOR TESTING PURPOSED ONLY
    # Move this to an update-able store of similar list for all galaxies!
    # These are the nonzero templates resulting from a ppxf of the full-
    # galaxy bin of NGC5557 using the full miles-massive template library.
    # This fit was not done new with the current massivepy library, but
    # taken from a (poorly-documented) previous fit dating from July 2014
fullMILES_1600newlist_newbins = [119,124,197,204,225,246,322,455,456,487,
                                 501,561,617,780,838,853,944,957,961]
fullMILES_1600newlist_oldbins = [38,99,136,225,246,252,322,376,409,455,
                                 501,548,561,617,780,838,853,944,956,957]
    # DO NOT KEEP HERE. these (both newbins and oldbins) were obtained with
    # the new code, and are store elsewhere also, they are only here for
    # testing while the code is updated to allow automatic reading of a 
    # file with this list in it.
float_tol = 10**(-10)
relaxed_tol = 10**(-4)

# pPXF conventions
ppxf_losvd_sampling_factor = 5
    # When convolving template spectra, pPXF samples losvds on the interval
    # central_velocity +- 5*sigma. This is stored here to allow an exact
    # reconstruction of the pPXF model.

# math
gaussian_fwhm_over_sigma = 2*np.sqrt(2*np.log(2))

# directory structure
#path_to_datamap = "etc/datamap.txt" # file giving locations of data stores
#Should not be needed anymore!

# bin number conventions
badfiber_bin_id = -666  #For fibers removed before binning step
unusedfiber_bin_id = -100  #For fibers not used in binning step

#fullgal_allfib_bin_id = 0
#fullgal_binfib_bin_id = -1
#fullgal_symfib_bin_id = -2

# template spectral type color conventions for plotting
# based on first character of 'spt' in miles catalog
spectype_colors = {'A':'aqua','B':'blue','G':'green','F':'lime','I':'indigo',
                   'M':'magenta','K':'crimson','-':'black','S':'orange',
                   '0':'gray','s':'tan','R':'yellow','H':'gold'}
