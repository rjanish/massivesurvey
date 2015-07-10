"""
This script processes the raw MILES stellar template library into a
new library that is more useful for spectral fitting.

The default library is not altered, rather a processed copy is made.
The changes are:
  - The star m0970V has been removed due to an instrumental or
    reduction artifact in the spectrum. There is a step-like feature
    at ~ 7000 A, redward of which the spectrum is identically 0.
    This feature drives pPXF to crash on huge model sigma values.
  - The stars m0742V and m0468V have been removed for similar reasons.
    (m0742V has 3 zero values at the end, m0468V has one in the middle.)
  - The index file as been renamed, pruned of unneeded information,
    and reformatted to be easily readable by python pandas.
  - One star in the raw index is numbered '022.' and there is no 0221
    listed, though the spectrum file m0221V exists. It must be a typo,
    which is corrected in the new index: '022.' -> '0221'
These changes are hard-coded into the first few lines of this script.

input:
  takes one command line argument, a path to the input parameter text file
  process_miles_params_example.txt is an example

output:
  processed MILES library, as a directory containing a catalog file, a
  library ir file, and a subdirectory of template spectra
"""


import os
import argparse
import shutil
import re

import pandas as pd

import utilities as utl
import massivepy.templates as temps
import massivepy.constants as const


BAD_MILES_TEMPLATES = [970, 742, 468]  # MILES stars to omit
MILESRAW_ID_MAP = {'022.':'0221'}  # correction of raw MILES index typo
MILESRAW_INDEX_PATTERN = (r"^(?P<object_name>.+?)\s+"
                          r"(?P<in_cluster>Cl)?\s+"
                          r"(?P<miles_id>0\d\d[\d\.])\s+"
                          r"(?P<cat_id>\d{3})?\s+"
                          r"(?P<spt>.*?)\s+"
                          r"(?P<teff>\d+|-+)\s+"
                          r"(?P<logg>\d.\d{2}|-+)\s+"
                          r"(?P<z>-?\d.\d{2}|-+)\s+"
                          r"(?:[\dA-Za-z]{1,3}|-+)\s+"  # matches references
                          r"(?P<libs>[A-Z\s]*?)\s*$")  # in other libraries
    # re to parse lines of raw MILES index - named groups' data will be saved
INDEX_CONVERSIONS = {"object_name":lambda s: utl.safe_str(s),
                     "in_cluster":lambda s: s == 'Cl',
                     "miles_id":lambda s: utl.safe_str(s),
                     "cat_id":lambda s: ('--' if s is None
                                              else utl.safe_str(s)),
                     "spt":lambda s: utl.safe_str(re.sub(r'\s+', '', s)),
                     "teff":lambda s: utl.safe_int(s),
                     "logg":lambda s: utl.safe_float(s),
                     "z":lambda s: utl.safe_float(s),
                     "libs":lambda s: utl.safe_str(re.sub(r'\s+', '', s))}
    # conversions to be applied to each column of raw MILES index file

# parse arguments (location of input param file)
parser = argparse.ArgumentParser(description=__doc__,
                formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("paramfile", type=str,
                    help="path to input parameter file for this script")
args = parser.parse_args()
paramfile_path = args.paramfile

# parse input parameter file
input_params = utl.read_dict_file(paramfile_path)
path_milesraw_spectra = input_params['path_milesraw_spectra']
path_milesraw_index = input_params['path_milesraw_index']
library_fwhm = input_params['library_fwhm']
path_new = input_params['path_new']
path_new_spectra = os.path.join(path_new, input_params['new_spectra_subdir'])
path_new_index = os.path.join(path_new, input_params['new_index_filename'])
path_new_ir = os.path.join(path_new, input_params['new_ir_filename'])

# make new library
print "making new library at {} ...".format(path_new)
os.mkdir(path_new)
os.mkdir(path_new_spectra)
old_spectra = utl.re_filesearch(".*", dir=path_milesraw_spectra)[0]
for old_template_path in old_spectra:
    old_template_filename = os.path.split(old_template_path)[1]
    #Skip hidden files (e.g. .DS_Store)
    if old_template_filename[0] == '.':
        continue
    old_template_id = temps.miles_filename_to_number(old_template_filename)
    if old_template_id in BAD_MILES_TEMPLATES:
        print "omitting template {}".format(old_template_id)
        continue
    shutil.copy(old_template_path, path_new_spectra)
print "done - copied {} templates".format(len(os.listdir(path_new_spectra)))
# read raw MILES index file
miles_ids, template_props = [], []
with open(path_milesraw_index, 'r') as milesraw_index:
    print "reading raw index file..."
    for line in milesraw_index:
        match = re.match(MILESRAW_INDEX_PATTERN, line)
        if match:
            line_contents = match.groupdict()
            string_id = line_contents['miles_id']
            if string_id in MILESRAW_ID_MAP:  # typo fix
                line_contents["miles_id"] = MILESRAW_ID_MAP[string_id]
            miles_id = int(line_contents["miles_id"])
            if miles_id in BAD_MILES_TEMPLATES:  # bad template fix
                print "omitting line for template {}".format(miles_id)
            else:
                for column, value in line_contents.iteritems():
                    line_contents[column] = INDEX_CONVERSIONS[column](value)
                template_props.append(line_contents)
                miles_ids.append(miles_id)
        elif len(template_props) > 0:  # no longer reading header
            raise Exception("MILESRAW_INDEX_PATTERN does not match all lines")
        else:
            print "skipping header line"
print "done - read {} lines".format(len(template_props))
new_index = pd.DataFrame(template_props, index=miles_ids)
  # setting index argument will force rows to sort by miles ID number
new_index.sort(axis=0, inplace=True)
new_index.to_csv(path_new_index, index=False)
print "saved new index at {}".format(path_new_index)
with open(path_new_ir, "w") as text_file:
    ir_header = ("# This the the Gaussian full-width at half-max of the "
                 "spectral resolution\n# of the MILES template library, "
                 "in Angstroms, as reported on the MILES website.\n")
    text_file.write(ir_header)
    text_file.write("library_fwhm\t{}".format(library_fwhm))
print "saved ir in file at {}".format(path_new_ir)
