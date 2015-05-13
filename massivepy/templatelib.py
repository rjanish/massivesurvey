""" Module for working with stellar template libraries """


# import os
# from re import match

# import numpy as np

# from pyutils import safe_float, safe_str


# miles_dir = "/stg/current/stellar-templates/miles-massive"
# miles_pattern = r"m\d{4}V"
# miles_index = "/stg/current/stellar-templates/miles-massive/miles-index.txt"


def miles_filename_conversion(num):
    """
    Converts an id number from the MILES stellar template library into
    matching default filename (tail only) that holds the spectrum.

    Example:
    > miles_filename_conversion(42)
    'm0042V'
    """
    return "m{:04d}V".format(int(num))


# def read_miles_index(index_path):
#     """
#     Read the MILES index file into a nested dictionary. The dictionary
#     is indexed by MILES template id number, and contains for each
#     template a dictionary of template star properties.

#     This function does *not* read the raw index file as downloaded
#     from the MILES website, but rather it read the index file as
#     prepared for use in the MASSIVE survey. This is a ascii file,
#     beginning with a header marked by '#' comment character and
#     following with a 6 column table containing one line for every star
#     in the library.  The columns from left to right are the object's
#     name, cluster star (boolean - listed as 'cl' if in a cluster or
#     blank if not), MILES number, spectral type, temperature, surface
#     gravity, and metallicity. The columns are delineated by whitespace
#     of varying amount.  Occasionally, the temperature, surface
#     gravity, or metallicity will be missing and given as '--', as will
#     the spectral type which is simply left blank. This function
#     requires that the above format is followed.

#     Args:
#     index_path - string
#         The default is the location of the MILES path on my thinkpad

#     Returns:
#     index - dict
#         A dictionary indexed by MILES id number, containing for each
#         template star a sub-dictionary of stellar properties.
#         Example: the index entry for MILES star 12 is:
#             >>> i[12]
#             {'cluster': False, 'logg': 5.19, 'obj': 'D001326B',
#              'spt': 'M6 V', 'teff': 3370.0, 'z': -1.4}
#         If logg, teff, or z are unknown, they are assigned NaN, and
#         if spt is unknown it is assigned the string '----'.
#     """
#     pattern = (r"^[^#](?P<obj>.+?)\s+"
#                r"(?P<cluster>Cl)?\s+"
#                r"0(?P<id>\d{3})\s+"
#                r"(?P<spt>.*?)\s+"
#                r"(?P<teff>\d+|--)\s+"
#                r"(?P<logg>\d.\d{2}|--)\s+"
#                r"(?P<z>-?\d.\d{2}|--)\s+$")
#         # hardcoded format of the MASSIVE-prepared MILES index file
#     conv = {"obj":str, "cluster":bool, "id":int, "spt":safe_str,
#             "teff":safe_float, "logg":safe_float, "z":safe_float}
#         # type conversion functions for each template property
#     index = {}
#     with open(index_path, 'r') as index_file:
#         for line in index_file:
#             matched_line = match(pattern, line)
#             if not matched_line:  # skip header and trailing blanks
#                 continue
#             template_props = matched_line.groupdict()
#             id = conv['id'](template_props['id'])
#             index[id] = {}
#             for key, value in template_props.iteritems():
#                 if key != 'id':
#                     index[id][key] = conv[key](value)
#     return index


# def get_miles_sublibrary(template_numbers):
#     """
#     Construct a library, with both spectra and a stellar properties
#     dictionary, using only a subset of the MILES library.

#     Args:
#     template_numbers - 1d arraylike
#         A list of template numbers to include in the sublibrary
#     """
#     full_index = read_miles_index(miles_index)
#     spectra, index = [], {}
#     for num in template_numbers:
#         try:
#             filename = miles_filename_conversion(num)
#             filepath = os.path.join(miles_dir, filename)
#             template = np.loadtxt(filepath)
#         except Exception, msg:
#             print "Skipping template {}\n{}".format(num, msg)
#             continue
#         spectra.append(template[:, 1])
#         index[num] = full_index[num]
#         wavelengths = template[:, 0]
#     return np.array(spectra), wavelengths, index
