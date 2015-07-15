"""
This script packages the 'old-style' NGC4552 data (as produced by the
vaccine | pipe1 | pipe2 pipeline and then binning using my old python
scripts) into a format usable by the current MASSVIE pPXF pipeline.
"""


import argparse
import os

import numpy as np

import utilities as utl
import massivepy.constants as const


# get cmd line arguments
parser = argparse.ArgumentParser(description=__doc__,
                formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--specfiles", nargs='*',
                    help="paths of spectra")
parser.add_argument("--irfiles", nargs='*',
                    help="paths of instrumental spectral resolution")
parser.add_argument("--bad", nargs='*',
                    help="wavelength regions ??-?? [A] of bad data")
args = parser.parse_args()

# read spectral data
spec_paths = sorted(args.specfiles)
ir_paths = sorted(args.irfiles)
print 'matching files...'
specs, noises, all_waves, irs = [], [], [], []
bin_ids = np.arange(len(spec_paths), dtype=int)
for bin_iter, (spec_path, ir_path) in enumerate(zip(spec_paths, ir_paths)):
    print ("bin{:02d}: {} {}"
           "".format(bin_iter, *map(os.path.basename, [spec_path, ir_path])))
    [[spec, noise]], [spec_header] = utl.fits_quickread(spec_path)
    ir = np.loadtxt(ir_path)
    waves, junk = utl.fits_getcoordarray(spec_header)
    specs.append(spec)
    noises.append(noise)
    all_waves.append(waves)
    irs.append(ir)
specs = np.asarray(specs, dtype=float)
noises = np.asarray(noises, dtype=float)
all_waves = np.asarray(all_waves, dtype=float)
wave_mismatch = all_waves.std(axis=0)/np.mean(all_waves, axis=0)
if np.max(wave_mismatch) < const.float_tol:
    common_waves = all_waves[0]
else:
    raise Exception("wavelengths do not match!")
irs = np.asarray(irs, dtype=float)
# find bad data
bad_regions = np.asarray([map(float, sec.split('-')) for sec in args.bad])
bad_data = []
for spec, wave in zip(specs, waves):
    bad_data.append(utl.in_union_of_intervals(waves, bad_regions))
bad_data = np.asarray(bad_data, dtype=bool)
# make hdu list