import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import glob

#plt.ion()

subdata = []
origdata = []
orig_wdata = []
#oneDspectrum = []

#subfile='Desktop/jjj4967pefsm_sub2.fits'
subfile='jjj5099pefsm_subtracted_a.fits'
equalorig='Desktop/jjj5099pefsm.fits'
unequalorig='Desktop/jjj5099pefsm_w.fits'
#unequalorig='Desktop/jjj4967pefsm_w.fits'
title = 'Desktop/1D_jjj4967pefsm_sub2.fits'

subdata.append(fits.getdata(subfile))
origdata.append(fits.getdata(equalorig))
orig_wdata.append(fits.getdata(unequalorig))

# Create empty arrays
difference = np.zeros_like(subdata[0][0])
orig = np.zeros_like(origdata[0][0])
orig_w = np.zeros_like(orig_wdata[0][0])

# Sum up all the rows to get a 1D array
for i in range(1933,1937):
  difference = difference + subdata[0][i]
  orig = orig + origdata[0][i]
  orig_w = orig_w + orig_wdata[0][i]

#total = difference[0:1039]
plt.plot(abs(difference), 'o')
#plt.plot(orig)
#plt.plot(orig_w)

plt.xlabel('Pixel')
plt.ylabel('Value')
plt.yscale('log')

plt.legend(['Difference', 'Equal', 'Unequal'], loc='upper left')

plt.show()
