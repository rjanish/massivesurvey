""" bin all Mitchell fibers into full-galaxy bin"""


import os
import re
import argparse

import utilities as utl
import massivepy.constants as const
import massivepy.IFUspectrum as ifu
import massivepy.spectrum as spec


# defaults
datamap = utl.read_dict_file(const.path_to_datamap)
proc_cube_dir = datamap["proc_mitchell_cubes"]
binned_dir = datamap["binned_mitchell"]

# get cmd line arguments
parser = argparse.ArgumentParser(description=__doc__,
                formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("cubes", nargs='*', type=str,
                    help="The processed Michell datacubes to bin, passed as "
                         "a filename in the proc cube directory")
parser.add_argument("--destination_dir", action="store",
                    type=str, nargs=1, default=binned_dir, # TO DO: check this usage below
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
    # get bin layout
    ifuset = ifu.read_mitchell_datacube(path)
    ngc_match = re.search(const.re_ngc, path)
    if ngc_match is None:
        raise RuntimeError("No galaxy name found for path {}".format(path))
    else:
        ngc_num = ngc_match.groups()[0]
    ngc_name = "NGC{}".format(ngc_num)
    # bin
    delta_lambda = (ifuset.spectrumset.spec_region[1] -
                    ifuset.spectrumset.spec_region[0])
    full_galaxy = ifuset.spectrumset.collapse(
                                  weight_func=spec.SpectrumSet.compute_flux,
                                  norm_func=spec.SpectrumSet.compute_flux,
                                  norm_value=delta_lambda, id=0)
    full_galaxy.comments["Binning"] = ("this spectrum is the coadditon "
                                       "of all fibers in the galaxy")
    bindesc = "full galaxy bin"
    full_galaxy.name = bindesc
    fibers_in_fullgalaxy = ifuset.spectrumset.ids.copy()
    output_filename = ("{}_{}.fits"
                       "".format(ngc_name, re.sub("\s*", "", bindesc)))
    output_path = os.path.join(binned_dir, output_filename)
    full_galaxy.write_to_fits(output_path)