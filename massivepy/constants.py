"""
Miscellaneous constants and unit definitions.

Some simple functions with no other obvious home are here too.
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
fiducial_wavelength_logscale = 0.000238884836314
    # this is what spectrumset.get_logscale() returns for Jenny's "normal" cubes
    # want to return to this after shifting dithers on oversampled cubes
    # (or any other weird sampling activity)

float_tol = 10**(-10)
relaxed_tol = 10**(-4)

# pPXF conventions
ppxf_losvd_sampling_factor = 5
    # When convolving template spectra, pPXF samples losvds on the interval
    # central_velocity +- 5*sigma. This is stored here to allow an exact
    # reconstruction of the pPXF model.

# math
gaussian_fwhm_over_sigma = 2*np.sqrt(2*np.log(2))

# bin number conventions
badfiber_bin_id = -666  #For fibers removed before binning step
unusedfiber_bin_id = -100  #For fibers not used in binning step

# template spectral type color conventions for plotting
# based on first character of 'spt' in miles catalog
spectype_colors = {'A':'aqua','B':'blue','G':'green','F':'lime','I':'indigo',
                   'M':'magenta','K':'crimson','-':'black','S':'orange',
                   '0':'gray','s':'tan','R':'yellow','H':'gold'}


# misc functions
def flat_plus_poisson(flux, flatnoise, fluxscale):
    return np.sqrt(flatnoise**2 + fluxscale*flux)

def arcsec_to_kpc(x, D):
    """convert x to kpc from arcsec, given D (distance along LOS)"""
    return D*np.pi*x/(3600*180)

def kpc_to_arcsec(x, D):
    """convert x to arcsec from kpc, given D (distance along LOS)"""
    return x*180*3600/(D*np.pi)

def re_conversion(re,d,mode='toNSA'):
    """converts Re between NSA and 2MASS; assumes arcsec as units.
    uses equation 2 from survey paper"""
    re_kpc = arcsec_to_kpc(re,d)
    if mode=='toNSA':
        re_new = 10**( (np.log10(re_kpc)+0.076) / 0.8 )
    elif mode=='fromNSA':
        re_new = 10**( 0.8*np.log10(re_kpc) - 0.076 )
    else:
        raise Exception('You broke it.')
    return kpc_to_arcsec(re_new,d)
