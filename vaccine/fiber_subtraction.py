# The purpose of this script is to take two frames and subtract their values from one fiber and plot the difference/fractional difference.

import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col


# Test image
image = 'Desktop/jjj5099pefsm_w.fits'

# Standard image
prev_image = 'Desktop/jjj5099pefsm.fits'

# Title for resulting frame
title = 'jjj5099pefsm_fractional_w.fits'

# Build arrays
image1 = []
image_base = []
fiber1 = []
fiber_base = []
difference = []
fractional = []

# Get data
image1.append(fits.getdata(image))
image_base.append(fits.getdata(prev_image))

# Create empty arrays
fiber1 = np.zeros_like(image1[0][0])
fiber_base = np.zeros_like(image1[0][0])
difference = np.zeros_like(image1[0][0])
fractional = np.zeros_like(image1[0][0])

# testing
print 'image1', len(image1), image1[0].shape
print 'image_base', len(image_base), image_base[0].shape
print 'fiber1', fiber1.shape
print 'fiber_base', fiber_base.shape

# Sum up all the rows to get a 1D array
for i in range(1933,1937):
	fiber1 = fiber1 + image1[0][i]
	fiber_base = fiber_base + image_base[0][i]

# Take difference
difference = fiber1 - fiber_base
#print difference

# testing
print 'difference', difference.shape

# Divide
fractional = difference/fiber_base
#print fractional


plt.plot(abs(fractional), '.')
#plt.plot(orig)
#plt.plot(orig_w)

plt.xlabel('Pixel')
plt.ylabel('Value')
plt.yscale('log')

plt.legend(['Fractional Difference equal v unequal', 'Equal', 'Unequal'], loc='upper left')

plt.show()
