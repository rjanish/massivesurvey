"""
Combines all fibers in an unbinned Mitchell datacube into one spectrum

input:
  takes one command line argument, a path to the input parameter text file
  bin_mitchell_fullgalaxies_params_example.txt is an example
  can take multiple parameter files if you want to process multiple galaxies
  (give one param file per galaxy)

output:
  One binned datacube per galaxy, containing 1 binned spectrum each.
"""


import os
import re
import argparse

import utilities as utl
import massivepy.constants as const
import massivepy.IFUspectrum as ifu
import massivepy.spectrum as spec


# get cmd line arguments
parser = argparse.ArgumentParser(description=__doc__,
                formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("paramfiles", nargs='*', type=str,
                    help="path(s) to input parameter file(s)")
args = parser.parse_args()
all_paramfile_paths = args.paramfiles


for paramfile_path in all_paramfile_paths:
    # parse input parameter file
    input_params = utl.read_dict_file(paramfile_path)
    proc_cube_path = input_params['proc_mitchell_cube']
    if ((not os.path.isfile(proc_cube_path)) 
        or (os.path.splitext(proc_cube_path)[-1] != ".fits")):
        raise ValueError("Invalid raw datacube path {}, "
                         "must be .fits file".format(proc_cube_path))
    destination_dir = input_params['destination_dir']
    if not os.path.isdir(destination_dir):
        raise ValueError("Invalid destination dir {}".format(destination_dir))

    # get bin layout
    ifuset = ifu.read_mitchell_datacube(proc_cube_path)
    ngc_match = re.search(const.re_ngc, proc_cube_path)
    if ngc_match is None:
        msg = "No galaxy name found for path {}".format(proc_cube_path)
        raise RuntimeError(msg)
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
    bindesc = "fullgalaxybin"
    full_galaxy.name = bindesc
    output_filename = ("{}-{}.fits"
                       "".format(ngc_name, re.sub("\s*", "", bindesc)))
    output_path = os.path.join(destination_dir, output_filename)
    full_galaxy.write_to_fits(output_path)

print 'You may ignore the weird underflow error, it is not important.'
