"""
This script parses a directory of raw Mitchell IFU frames in .fits
format and compiles a list of frame types (science, arc, etc.).
"""


import argparse
import os
import sys
import collections

import utilities as utl


parser = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("data_dir", help="path to raw data directory")
args = parser.parse_args()
data_dir = args.data_dir
if not os.path.isdir(data_dir):
    raise ValueError("invalid data directory: {}".format(data_dir))

raw_fits, matches = utl.re_filesearch(r"(?P<frame_name>.*)\.fits", data_dir)
frame_names = [m.groups()[0] for m in matches]
make_defaultdict_list = lambda : collections.defaultdict(list)
frames_by_datetype = collections.defaultdict(make_defaultdict_list)
for path, name in zip(raw_fits, frame_names):
    [data], [header] = utl.fits_quickread(path)
    frame_type = header["OBJECT"]
    date = header["DATE-OBS"]
    frames_by_datetype[date][frame_type].append(name)
# for date, date_dict in frames_by_datetype.iteritems():
#     for frame_type, frames in date_dict.iteritems():
#         frames_by_datetype[date][frame_type] = sorted(frames)

# sys.stdout.write("frames from {}:\n".format(data_dir))
# for date in sorted(frames_by_datetype.keys()):
#     sys.stdout.write("\n{}:\n".format(date))
#     date_dict = frames_by_datetype[name]
#     for frame_type in sorted(date_dict.keys()):
#         sys.stdout.write("  {}:\n".format(frame_type))
#         frames = date_dict[frame_type]
#         for frame in frames:
#             sys.stdout.write("    {}\n".format(frame))
