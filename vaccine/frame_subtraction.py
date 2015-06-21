# The purpose of this script is to take two frames and subtract their values from each other to check that vaccine is properly working.

import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col

plt.ion()

# Test image
my_image = 'jjj5405pefsm.fits'

# Standard image
prev_image = 'jjj5405pefsm_a.fits'

# Title for resulting frame
title = 'jjj5405pefsm_fractional.fits'

# Get values
my_data = fits.getdata(my_image)
prev_data = fits.getdata(prev_image)

# Subtract images
SubImage = my_data - prev_data

# Fractional difference
#fractional = SubImage/prev_data

#plt.imshow(SubImage[::-1], cmap='gray', origin='lower')
#plt.show()

#Save fits file
hdu = fits.PrimaryHDU(SubImage)
hdu.writeto(title)
