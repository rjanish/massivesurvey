""" Miscellaneous useful functions for file handling tasks """


import os
import re
import pickle

import astropy.io.fits as fits
import numpy as np


def cleanpath(path, default=os.curdir, verbose=False, whine=False):
    """
    Returns the pathname purged of unneeded symbols, and possibly
    returns a default pathname if an invalid path is passed.
    
    Args:
    path - string
        The path to clean
    default - string, default=os.curdir
        The path to use if the passed path is invalid
    verbose - bool, default=False
        If True, print a warning message if the default path is used
    whine - bool, default=False
        If True, do not use the default path in place of an
        invalid passed path, rather raise an exception
    
    Returns: cleaned path

    Examples:
    > cleanpath("./dir/")
    "dir"

    > cleanpath(None)
    "."

    > cleanpath(None, whine=True)
    AttributeError: ...

    > cleanpath(None, verbose=True)
    Invalid path: None
    Returning default: .
    "."
    """
    if whine:
        return os.path.normpath(path)
    else:
        try:
            return os.path.normpath(path)
        except:  # passed path is not a valid filepath
            if verbose:
                print ("Invalid path: {}\n"
                       "Returning default: {}".format(path, default))
            return default


def re_filesearch(pattern, dir=None):
    """
    Lists all files in a directory that match a given regular
    expression, and gives their match results, including group vales.
    Only the filename and extension, not the full path, is considered.

    Args:
    pattern - string
        Regular expression pattern that matched files must satisfy.
    dir - string, default=current directory
        Path of directory in which to search for files.

    Returns: files, matches
    files - list of strings
        filenames of matched files, includes full path of the matched
        files if they are not located in the current directory
    matches - list of python re match objects
        re match result for each matched file, ordered as in 'files'
    """
    dir = cleanpath(dir)  # if invalid dir (e.g., None), gives default of '.'
    files_present = sorted(os.listdir(dir))  # gives only filename.ext
    matching_files, match_objects = [], []
    for filename in files_present:
        match_result = re.match(pattern, filename)
        if match_result:
            matching_files.append(os.path.join(dir, filename))
            match_objects.append(match_result)
    return matching_files, match_objects


def load_pickle(filename):
    """ Read a python pickle file """
    with open(filename, 'rb') as jar:
        obj = pickle.load(jar)
    return obj


def save_pickle(obj, filename):
    """ Save passed object into a python pickle file """
    with open(filename, 'wb') as jar:
        pickle.dump(obj, jar)
    return obj


def read_dict_file(filepath, delimiter=None, comment='#', whine=False,
                   conversions=[float, str], specific={}):
    """
    Parse a two column text file into a dictionary, with the first
    column becoming the keys and the second determining the values. A
    key appearing more than once in the file will be paired with a
    list containing all of its values arranged in order of appearance.

    Args:
    filepath - string
        Path to the file to be parsed.
        File format: The file must have two columns separated by the
        passed delimiter. The first delimiter encountered on each line
        marks the column division, while subsequent delimiters are
        part of the value data. Leading and trailing whitespace is
        stripped from each line and blank lines will be skipped.
    delimiter - string, default is any whitespace
        String separating keys and values
    comment - string, default='#'
       Everything after a comment character on a line is ignored
    conversions - list, default = [float, str]
        A list of functions mapping strings to some python object, to
        be applied to each value before storing in the output dict.
        Conversions are tried in order until success.
    specific - dict, no default
        A dict mapping keys in the file to a conversion function. If
        a key appearing in the file has an entry in specific, the
        function given in specific is used for conversion.
    whine - bool, default = True
        If True, a line that had no successful conversion attempts
        will throw an exception, otherwise it is skipped silently.

    Returns - output_dict
    output_dict - dictionary with all key and converted value pairs
    """
    output_dict = {}
    number_encountered = {}
    with open(filepath, 'r') as infofile:
        for line in infofile:
            comment_location = line.find(comment)  # is -1 if nothing found
            comment_present = (comment_location != -1)
            if comment_present:
                line = line[:comment_location]
            line = line.strip()
            if len(line) == 0:  # line is blank
                continue
            label, raw_data = line.split(delimiter, 1)
                # split(X, Y) splits on X at most Y times, starting at left
                # X = None splits on any whitespace
            # convert string entry
            if label in specific:
                conv_data = specific[label](raw_data)
            else:
                for conversion_func in conversions:
                    try:
                        conv_data = conversion_func(raw_data)
                        break # conversion success
                    except:
                        pass  # conversion failed
                else:
                    if whine:
                        raise ValueError("All conversions failed.")
            # update dictionary
            if label in number_encountered:
                number_encountered[label] += 1
            else:
                number_encountered[label] = 1
            if number_encountered[label] == 1:
                output_dict[label] = conv_data
            elif number_encountered[label] == 2:
                output_dict[label] = [output_dict[label], conv_data]
            elif number_encountered[label] >= 3:
                output_dict[label].append(conv_data)
    return output_dict


def fits_quickread(path):
    """
    A convenience function for accessing fits files.  This mostly
    serves to eliminate the 'fit.open ... fits.close' boilerplate.

    Args:
    path - string
        Path to a fits file

    Returns: fitslist
    fitslist - 2-element list
        This output list contains the fits data in fitslist[0] and the
        headers in fitslist[1].  Each entry in fitslist is itself a
        tuple containing the data array or header for a single
        extension.  I.e., the fitslist[0][n] contains the data (as a
        numpy array) in the nth fits extension and fitslist[1][n] is
        the nth extension's header (as an astropy fits header object).
        The 'primary' fits extension corresponds to n = 0.

    """
    hdu = fits.open(path, memmap=False)
        # memmap causes multiple file handles to be created, only one
        # of which is closed by the call to .close below, so using
        # True here results in rouge open file handles.  This is
        # probably a BUG in astropy.io.fits
    fitslist = [[ext.data, ext.header] for ext in hdu]
    hdu.close()
    fitslist = zip(*fitslist) # this executes a transpose on lists
    return fitslist


def fits_getcoordarray(header):
    """
    Returns the array of coordinate values corresponding to the data
    in a single .fits extension, as specified by the header keywords
    CRVAL, CDELT, etc.

    Args:
    header - fits Header object, or anything indexable as a dict
        This is the header specifying the coordinate system. It must
        include the key NAXIS and the three keys NAXISn, CRVALn, and
        CDELTn for each axis in the data.

    Returns: coords
    coords - list
        A list of arrays, with each array giving the coordinate values
        along an axis of the .fits data. The list is ordered according
        to the axis numbering of the .fits header.
    """
    coords = []
    num_axes = header["NAXIS"]
    for axis_index in xrange(num_axes):
        axis_number = axis_index + 1  # .fits headers' number axes from 1
        num_datapts = header["NAXIS{}".format(axis_number)]
        start_value = header["CRVAL{}".format(axis_number)]
        step = header["CDELT{}".format(axis_number)]
        coords.append(start_value + step*np.arange(num_datapts))
    return coords