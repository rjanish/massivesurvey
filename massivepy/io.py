"""
Basic repetiitve io tasks
"""

import os
import re

import utilities as utl
import massivepy.constants as const

def parse_paramfile_path(path):
    """
    Returns the directory and galaxy name, given a parameter file path.
    Checks that the first 7 characters of the file name match 'NGC####'.
    Also checks that the directory has #### somewhere in it.
    """
    output_dir = os.path.dirname(path)
    gal_name = os.path.basename(path)[0:7]
    if re.match(utl.force_full_match(const.re_ngc),gal_name) is None:
        raise Exception("Invalid galaxy name in parameter file path.")
    re_gal = re.search(const.re_ngc,output_dir)
    if re_gal is None:
        print "\nWarning, your output directory has no ngc name."
        print "Your organization is bad, go fix it.\n"
    elif re_gal.groups()[0] != gal_name[-4:]: #Test that galaxy number matches
        print "\nWarning, your output directory does not match your galaxy."
        print "Putting {} in {} directory\n".format(gal_name,re_gal.groups()[0])
    else:
        pass
    return output_dir, gal_name
