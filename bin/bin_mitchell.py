""" bin mitchell fibers """


import os
import re
import argparse
import functools
import pickle

import numpy as np
import pandas as pd

import utilities as utl
import massivepy.constants as const
import massivepy.IFUspectrum as ifu
import massivepy.spectrum as spec
import massivepy.binning as binning


# defaults
datamap = utl.read_dict_file(const.path_to_datamap)
proc_cube_dir = datamap["proc_mitchell_cubes"]
binned_dir = datamap["binned_mitchell"]
target_positions = pd.read_csv(datamap["target_positions"],
                               comment='#', sep="[ \t]+")

# get cmd line arguments
parser = argparse.ArgumentParser(description=__doc__,
                formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("cubes", nargs='*', type=str,
                    help="The processed Michell datacubes to bin, passed as "
                         "a filename in the proc cube directory")
parser.add_argument("--destination_dir", action="store",
                    type=str, nargs=1, default=proc_cube_dir,
                    help="Directory in which to place processed cubes")
args = parser.parse_args()
cube_paths = [os.path.normpath(os.path.join(proc_cube_dir, p))
              for p in args.cubes]
for path in cube_paths:
    if (not os.path.isfile(path)) or (os.path.splitext(path)[-1] != ".fits"):
        raise ValueError("Invalid raw datacube path {}, "
                         "must be .fits file".format(path))
dest_dir = os.path.normpath(args.destination_dir)
if not os.path.isdir(dest_dir):
    raise ValueError("Invalid destination dir {}".format(dest_dir))

for path in cube_paths:
    ifuset = ifu.read_mitchell_datacube(path)
    ngc_match = re.search(const.re_ngc, path)
    if ngc_match is None:
        raise RuntimeError("No galaxy name found for path {}".format(path))
    else:
        ngc_num = ngc_match.groups()[0]
    ngc_name = "NGC{}".format(ngc_num)
    gal_position = target_positions[target_positions.Name == ngc_name]
    gal_pa = gal_position.PA_best.iat[0]
    ma_bin = np.pi/2 - np.deg2rad(gal_pa)
    ma_xy = np.pi/2 + np.deg2rad(gal_pa)
    fiber_radius = const.mitchell_fiber_radius.value
    folded = functools.partial(binning.partition_quadparity_folded,
                               major_axis=ma_bin, aspect_ratio=1.5)
    binning_func = functools.partial(binning.polar_threshold_binning,
                                     angle_partition_func=folded)
    delta_lambda = (ifuset.spectrumset.spec_region[1] -
                    ifuset.spectrumset.spec_region[0])
    combine_func = functools.partial(spec.SpectrumSet.collapse, id=0,
                                     weight_func=spec.SpectrumSet.compute_flux,
                                     norm_func=spec.SpectrumSet.compute_flux,
                                     norm_value=delta_lambda)
    binned = ifuset.s2n_spacial_binning(binning_func=binning_func,
                                        combine_func=combine_func,
                                        threshold=20)
    grouped_ids, radial_bounds, angular_bounds = binned
    # results
    fiber_ids = ifuset.spectrumset.ids
    single_fiber_bins = [l for l in grouped_ids if len(l) == 1]
    flat_binned_fibers = [f for l in grouped_ids for f in l]
    unbinned_fibers = [f for f in fiber_ids if f not in flat_binned_fibers]
    print "{} total number of bins".format(len(grouped_ids))
    print "{} single-fiber bins".format(len(single_fiber_bins))
    print "{} un-binned outer fibers".format(len(unbinned_fibers))
    print "multi-fiber bin layout:"
    for iter, [(rin, rout), angles] in enumerate(zip(radial_bounds,
                                                     angular_bounds)):
        print ("  {:2d}: radius {:4.1f} to {:4.1f}, {} angular bins"
               "".format(iter + 1, rin, rout, len(angles)))
    bindesc = "polar_folded_s2n20"
    output_base = os.path.join(binned_dir, "{}_{}".format(ngc_name, bindesc))
    # binned_data_filename = "{}.fits".format(output_base)
    # output_path = os.path.join(binned_dir, binned_data_filename)
    # binned.write_to_fits(output_path)
    with open("{}_binned_fibers.p".format(output_base), 'wb') as pfile:
        pickle.dump(grouped_ids, pfile)
    with open("{}_unbinned_fibers.p".format(output_base), 'wb') as pfile:
        pickle.dump(unbinned_fibers, pfile)
    with open("{}_radial_bounds.p".format(output_base), 'wb') as pfile:
        pickle.dump(radial_bounds, pfile)
    for anulus_iter, angle_bounds in enumerate(angular_bounds):
        filename = "{}_angular_bounds-{}.p".format(output_base, anulus_iter)
        with open(filename, 'wb') as pfile:
            pickle.dump(angle_bounds, pfile)
