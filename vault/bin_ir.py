""" bin ir measurments """


import sys
import os

import numpy as np

from fileutils import load_pickle

galaxy_name = sys.argv[1]
ir_filename = sys.argv[2]
bindata_filename = sys.argv[3]
target_dir = sys.argv[4]

ir = load_pickle(ir_filename)
bindata = load_pickle(bindata_filename)
fibers_in_bin = [pair[0] for pair in bindata[0]]
fiber_flux = bindata[3]
# add full-galaxy bin
all_fibers =  [f for l in fibers_in_bin for f in l]
fibers_in_bin = [all_fibers] + fibers_in_bin
for bin_number, fibers in enumerate(fibers_in_bin):
    target_ir = ir[:, fibers, :]
    target_fluxes = fiber_flux[fibers]
    weights = target_fluxes/np.sum(target_fluxes)
    binned_fwhm = np.sum(target_ir[1, :, :].T*weights, axis=1)
    binned_centers = np.sum(target_ir[0, :, :].T*weights, axis=1)
    binned_ir = np.array([binned_centers, binned_fwhm]).T
    header = ("bin{:02d} instrumental resolution\n"
              "computed as flux-weighed mean over fiber fwhm,\n"
              "with fiber fwhm from a gaussian fit to arc frames\n"
              "col 1: wavelength, A\n"
              "col 2: gaussian fwhm, A".format(bin_number))
    filename = "bin{:02d}_{}_ir.txt".format(bin_number, galaxy_name)
    path = os.path.join(target_dir, filename)
    np.savetxt(path, binned_ir, fmt=['%7.2f', '%5.3f'],
               header=header, delimiter='  ')