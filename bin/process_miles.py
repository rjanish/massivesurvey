"""
This script processes the raw MILES stellar template library into a
new library that is more useful for spectral fitting.

The default library is not altered, rather a processed copy is made.
The changes are:
  - The star m0790V has been removed due to an instrumental or
    reduction artifact in the spectrum. There is a step-like feature
    at ~ 7000 A, redward of which the spectrum is identically 0.
    This feature drives pPXF to crash on huge model sigma values.
  - The index file as been renamed, pruned of unneeded information,
    and reformatted to be easily readable by python pandas.
  - One star in the raw index is numbered '022.' and there is no 0221
    listed, though the spectrum file m0221V exists. It must be a typo,
    which is corrected in the new index: '022.' -> '0221'
These changes are hard-coded into the first few lines of this script.
"""


import os
import argparse
import shutil
import re

import pandas as pd

import utilities as utl
import massivepy.templates as temps
import massivepy.constants as const


# defaults
DATAPATHS_PATH = const.path_to_datamap
    # file specifying locations of MASSIVE static data
TEMPLATE_LIBS_KEY = 'template_libraries'  # Entry needed from DATAPATHS_PATH

MILESRAW_LIBNAME = 'miles-raw'  # directory where MILES has been downloaded
MILESRAW_INDEX_FILENAME = 'paramsMILES_v9.1.txt'  # raw MILES template index
MILESRAW_SPECTRA_SUBDIR = 'spectra' # location of raw MILES spectra
MILESRAW_README_FILENAME = 'README.txt' # location of raw MILES spectra
BAD_MILES_TEMPLATES = [790]  # MILES stars to omit
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
NEW_INDEX_FILENAME = 'catalog.txt' # new template index
NEW_SPECTRA_SUBDIR = 'spectra' # location of raw MILES spectra

# parse destination name
parser = argparse.ArgumentParser(description=__doc__,
                formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("new_libname", type=str,
                    help="name of the new processed library")
args = parser.parse_args()
new_libname = args.new_libname
# get raw MILES library locations
data_paths = utl.read_dict_file(DATAPATHS_PATH)
template_libs_path = data_paths[TEMPLATE_LIBS_KEY]
libraries = [name for name in os.listdir(template_libs_path)
             if os.path.isdir(os.path.join(template_libs_path, name))]
path_milesraw = os.path.join(template_libs_path, MILESRAW_LIBNAME)
path_milesraw_spectra = os.path.join(path_milesraw, MILESRAW_SPECTRA_SUBDIR)
path_milesraw_index = os.path.join(path_milesraw, MILESRAW_INDEX_FILENAME)
# prune raw MILES into new library
# if new_libname in libraries:
#     raise ValueError("library name {} already in use".format(new_libname))
path_new = os.path.join(template_libs_path, new_libname)
path_new_spectra = os.path.join(path_new, NEW_SPECTRA_SUBDIR)
path_new_index = os.path.join(path_new, NEW_INDEX_FILENAME)
print "making new library at {} ...".format(path_new)
os.mkdir(path_new)
os.mkdir(path_new_spectra)
old_spectra = utl.re_filesearch(".*", dir=path_milesraw_spectra)[0]
for old_template_path in old_spectra:
    old_template_filename = os.path.split(old_template_path)[1]
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
path_milesraw_readme = os.path.join(path_milesraw, MILESRAW_README_FILENAME)
path_new_readme = os.path.join(path_new,
                               MILESRAW_README_FILENAME + "-OUTDATED")
shutil.copy(path_milesraw_readme, path_new_readme)
print "README copied to {}".format(path_new_readme)
print "README is outdated - must update by hand!"