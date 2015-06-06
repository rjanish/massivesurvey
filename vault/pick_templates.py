
import sys
sys.path.append("/stg/current/massivetesting/lib")
import shutil
import os

import numpy as np

import utilities as utl
import templatelib


fit_dir, temp_dir, mult_degree, output_base = sys.argv[1:5]
bins = map(int, sys.argv[5:])

temp_pat = r"m\d{4}V"
index_pat = r".*index.*?\.(?:dat|txt)"
templates_filename = utl.re_filesearch(temp_pat, temp_dir)[0]
templates_filename.sort()
templates = np.array([int(os.path.split(fl)[-1][1:-1])
                      for fl in templates_filename])
template_index_filename = utl.re_filesearch(index_pat, temp_dir)[0]
if len(template_index_filename) == 1:
    template_index_filename = template_index_filename[0]
else:
    raise Exception("More than 1 template index found!")
template_index = templatelib.read_miles_index(template_index_filename)

tempweight_pat = (r".*?-{}-tempweights.txt".format(mult_degree))
w_filename = utl.re_filesearch(tempweight_pat, fit_dir)[0]
if len(w_filename) != 1:
    raise Exception("More than 1 template weight file found!")
w_filename = w_filename[0]
weights = np.loadtxt(w_filename)

nonzero_templates = {}
for bin_num in bins:
    bin_weights = weights[:, bin_num]
    nonzero_templates[bin_num] = templates[bin_weights > 0]
    num_nonzero = nonzero_templates[bin_num].size
    total_num = bin_weights.size
    bin_header =  ("bin {:02d}: {}/{} nonzero templates"
                   "".format(bin_num, num_nonzero, total_num))
    print bin_header
    for nonzero_temp in nonzero_templates[bin_num]:
        print ("{0:3d}  {1[obj]:12s}  {1[spt]:12s}  {1[teff]:6.1f}  {1[z]:5.2f}"
               "".format(nonzero_temp, template_index[nonzero_temp]))
    print
    output_dir = "{}-{:02d}".format(output_base, bin_num)
    try:
        os.mkdir(output_dir)
    except:
        pass
    for nonzero_temp in nonzero_templates[bin_num]:
        nonzero_filename = "m{:04d}V".format(nonzero_temp)
        nonzero_path = os.path.join(temp_dir, nonzero_filename)
        shutil.copy(nonzero_path, output_dir)
    shutil.copy(template_index_filename, output_dir)





