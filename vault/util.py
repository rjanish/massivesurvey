"""
This module holds miscellaneous functions that have use beyond ifu
analysis. The functions are mostly logistical or purely mathematical.
"""

import re
import os

import numpy as np


def gaussian(x, mu, sigma):
    """
    Returns the normalized element-wise Gaussian G(x) for an array x,
    according to:
        $$ G(x) = \frac{1}{\sigma \sqrt{2 \pi}}
           e^{-\frac{1}{2} \left( \frac{x - \mu}{\sigma} \right)^2} $$
    """
    arg = (x - mu)/sigma
    norm = sigma*np.sqrt(2*np.pi)
    return np.exp(-0.5*(arg**2))/norm


def principal_value_shift(data, lower=0.0, period=2*np.pi):
    """
    Shift data by multiples of period until lower <= data < period.
    """
    data = np.array(data)
    while np.sum(data < lower) > 0:
       data[data < lower] += period
    upper = lower + period
    while np.sum(data >= upper) > 0:
       data[data >= upper] -= period
    return data


def read_dict_file(filename, delimiter=None, comment='#', skip=False,
                   conversions=[float, str], specific={}):
    """
    Parse a two column text file into a dictionary, with the first
    column becoming the keys and the second the values.  If a key
    appears more than once in the file, then its value in the
    dictionary will be a list of all values given in the file arranged
    by order of appearance. Values are read as floats, and then
    strings, though custom conversion can be specified.

    Args:
    filename - string
        Name of file to be parsed. The file must contain two columns
        separated by the passed delimiter. The first delimiter
        encountered on each line will mark the column division,
        subsequent delimiters are treated as part of the value data.
        Leading and trailing whitespace is stripped from each line.
        Blank lines will be skipped.
    delimiter - string, default is any whitespace
        String separating keys and values
    comment - string, default='#'
       Anything after a comment character on a line is ignored
    conversions - list, default = [float, str]
        A list of functions mapping strings to some python object, to
        be applied to each value before storing in the output dict.
        Conversions are tried in order until success.
    specific - dict, no default
        A dict mapping keys in the file to conversion functions. When
        a key appearing in the file has an entry in specific, the
        function given in specific is used first before attempting
        the functions given in conversions.
    skip - bool, default = False
        If False, a line that had no successful conversion attempts
        will throw an exception, otherwise it is skipped silently.

    Returns - file_dict
    file_dict - dictionary with all key and converted value pairs
    """
    max_splits = 1
    file_dict = {}
    number_encountered = {}
    with open(filename, 'r') as infofile:
        for line in infofile:
            comment_location = line.find(comment)  # is -1 if nothing found
            comment_present = (comment_location != -1)
            if comment_present:
                line = line[:comment_location]
            if len(line) == 0:
                continue
            label, raw_data = line.strip().split(delimiter, max_splits)
                # split(None, X) splits on any whitespace
            try:
                conv_data = specific[label](raw_data)
            except (KeyError, ValueError):
                # no specific conversion function given or conversion failed
                for conversion_func in conversions:
                    try:
                        conv_data = conversion_func(raw_data)
                    except ValueError: # conversion failed
                        pass
                    else: # conversion success
                        break
                else:
                    if not skip:
                        raise ValueError("All conversions failed.")
            if label in number_encountered:
                number_encountered[label] += 1
            else:
                number_encountered[label] = 1
            if number_encountered[label] == 1:
                file_dict[label] = conv_data
            elif number_encountered[label] == 2:
                file_dict[label] = [file_dict[label], conv_data]
            elif number_encountered[label] >= 3:
                file_dict[label].append(conv_data)
    return file_dict


def read_target_info(filename):
    """
    Parse the target info file into dictionary. See readme.txt for
    the assumed formating of the target info file.
    """
    as_string = ["name", "fiber_data", "data_dir"]
    as_float = ["center_ra", "center_dec", "pa", "fiber_radius"]
    as_int = ["first_fiber", "last_fiber"]
    conversions = [(as_string, str), (as_int, int), (as_float, float)]
        # for regular expression pattern, when read from a file any control
        # characters are automatically escaped, so converting straight to
        # string is fine, no raw string nonsense is needed
    return read_dict_file(filename, conversions)


def read_fitting_info(filename):
    """
    Parse the target info file into dictionary. See readme.txt for
    the assumed formating of the target info file.
    """
    as_string = ["results_dir", "intermediates_dir", "template_dir",
                 "template_index", "fit_by"]
    as_float = ["template_fwhm", "bias", "v_guess", "sigma_guess", "hn_guess"]
    as_int = ["additive_degree", "multiplicative_degree", "moments_to_fit"]
    as_range = ["mask_observed", "mask_rest"]
        # format is ##-##, convert to [##, ##] pair representing an interval
    conversions = [(as_string, str), (as_int, int), (as_float, float),
                   (as_range, lambda s: map(float, s.split('-')))]
    return read_dict_file(filename, conversions)


def re_filesearch(pattern, directory=None):
    """
    Return a list of all files in passed directory that match the
    passed regular expression.  Only the filename, not the full
    pathnames are considering for matching, however the returned
    matches will include pathnames if a directory is specified.

    Args:
    pattern - string, regular expression pattern
    dir - string, directory name, default is current directory

    Returns: files, matches
    files - list of strings, matching file paths including directory.
    matches - list of python re match objects for each matched file.
    """
    if directory is None:
        directory = os.curdir
    files_present = os.listdir(directory)
    matching_files, match_objects = [], []
    for filename in files_present:
        match_result = re.match(pattern, filename)
        if match_result:
            matching_files.append("{}/{}".format(directory, filename))
            match_objects.append(match_result)
    return matching_files, match_objects


def cartesian_product(arrays, output_dtype=None):
    """
    Generate a Cartesian product of input arrays.

    Args:
    arrays - list
        List of arrays with which to form the Cartesian
        product.  Input arrays will be treated as 1D.
    output_dtype - numpy data type
        The dtype of the output array. If unspecified, will be
        taken to match the dtype of the first  element of arrays.

    Returns:
    output - 2D array
        Array of shape (arrays[0].size*arrays[1].size..., len(arrays))
        containing the Cartesian product of the input arrays.

    Examples:
    >>> cartesian_product(([1, 2, 3, 4], [5.0], [6.0, 7.0]))
    array([[1, 5, 6],
           [1, 5, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 5, 6],
           [3, 5, 7],
           [4, 5, 6],
           [4, 5, 7]])
    >>> cartesian_product([['boil', 'mash', 'stew'],
                           ['potatoes', 'rabbit']], dtype='|S8')
    array([['boil', 'potatoes'],
           ['boil', 'rabbit'],
           ['mash', 'potatoes'],
           ['mash', 'rabbit'],
           ['stew', 'potatoes'],
           ['stew', 'rabbit']], dtype='|S8')
    """
    # make output container
    arrays = [np.array(x).flatten() for x in arrays]
    if output_dtype is None:
        output_dtype = arrays[0].dtype
    num_output_elements = np.prod([x.size for x in arrays])
    output_element_size = len(arrays)
    output_shape = (num_output_elements, output_element_size)
    output = np.zeros(output_shape, dtype=output_dtype)
    # form Cartesian product
    repetitions = num_output_elements/arrays[0].size
        # the number of times that each element of arrays[0] will
        # appear as the first entry in an element of the output
    output[:, 0] = np.repeat(arrays[0], repetitions)
        # for each block of output elements with identical first
        # entry, the remaining pattern of entries within each block
        # will be identical to that of any other block and is just the
        # Cartesian produced of the remaining arrays: recursion!
    arrays_remaining = bool(arrays[1:])
    if arrays_remaining:
        sub_output = cartesian_product(arrays[1:], output_dtype=output_dtype)
        for block_number in xrange(arrays[0].size):
            block_start = block_number*repetitions
            block_end = block_start + repetitions
            output[block_start:block_end, 1:] = sub_output
    return output


def safe_int(value, fail_value=np.nan):
    """ Return value converted to int or fail_value """
    try:
        return int(value)
    except:
        return fail_value


def safe_str(value, fail_value='----'):
    """ Return value converted to string or fail_value """
    try:
        to_string = str(value)
        if to_string:
            return to_string
        else:
            return fail_value
    except:
        return fail_value


def safe_float(value, fail_value=np.nan):
    """ Return value converted to float or fail_value """
    try:
        return float(value)
    except:
        return fail_value


def in_union_of_intervals(values, ranges):
    """
    Test if values are contained in a union of inclusive 1D intervals.

    Args:
    values - arraylike
        The values to be tested for inclusion.
    ranges - iterable
        An iterable of 1D inclusive intervals, each interval specified
        as a range [lower, upper].

    Returns: in_union
    in_union - bool array, shape matching values
        True for the elements in values that are contained in the
        union of the intervals in ranges.
    """
    in_union = np.zeros(values.shape, dtype=np.bool)
    for lower, upper in ranges:
        in_interval = (lower <= values) & (values <= upper)
        in_union = in_union | in_interval
    return in_union


def inflate(vector, direction, num):
    """
    Copy the passed vector to fill out a 2D array with either all
    columns or all rows identical to the passed vector.

    Args:
    vector - 1d arraylike
        Vector to be inflated into a 2d array.
    direction - string indicator, either 'h' or 'v'
        Specifies whether the passed vector should be considered a
        row or column vector.  Options are:
        'h' - row vector input, the output is created by using the
              passed vector as repeating matrix rows
        'v' - columns vector input, the output is created by using the
              passed vector as repeating matrix columns
    num - int
        The number of copies to make of the passed vector.  For
        direction 'h', this is the number of rows in the output, and
        for direction 'v' it is the number of columns.

    Return: matrix
    matrix - 2d ndarray
        Matrix composed of copies of the input vector

    Examples:
    v = np.arange(4)
    inflate(v, 'h', 2)
        [[0, 1, 2, 3],
         [0, 1, 2, 3]]
    inflate(v, 'v', 6)
        [[0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1],
         [2, 2, 2, 2, 2, 2],
         [3, 3, 3, 3, 3, 3]]
    """
    if direction == 'h':
        return np.vstack([vector]*num)
    elif direction == 'v':
        return np.hstack([np.transpose([vector])]*num)
    else:
        raise Exception("direction must be either 'h' or 'v'")


def compute_projected_confidences(covariance, fraction=0.683):
    """
    Given a covariance matrix describing assumed Gaussian distribution
    in D dimensions, this computes the D-ellipsoid containing the
    given probability fraction and projects that ellipsoid onto the
    1D directions defined by each individual parameter.

    Args:
    covariance - 2D arraylike
        The covariance matrix
    fraction - float, default=0.683
        The fraction of probability weight to be enclosed by the
        D-ellipsoid.  Default is 0.683, but this does not quite give
        a '1-sigma' ellipsoid: the weight enclosed by the covariance
        ellipsoid of a Gaussian distribution depends on dimension and
        is decreasing.  In 1D, 1 sigma corresponds to 68.3%
        confidence, but in higher dimension 1 sigma encloses less than
        68.3% of the probability weight.  This code uses percentiles
        rather than sigma, so the ellipsoid returned is in general
        going to be some larger multiple of the 1 sigma ellipse than
        would be naively expected from the 1D case.

    Returns: intervals
    intervals - 1D arraylike
        The half-width of the projected confidence intervals, given
        in the same order as the rows of the passed covariance matrix.
    """
    cov_metric = np.linalg.inv(covariance)
        # Mahalanobis metric corresponding to the data's covariance matrix
    center = np.median(prob_draws, axis=0)
    mdist_sq = []  # squared Mahalanobis distance of each draw
    for draw in prob_draws:
        shifted = draw - center
        mdist_sq.append(np.dot(shifted.T, np.dot(cov_metric, shifted)))
    percent = fraction*100
    conf_mdist_sq = np.percentile(mdist_sq, percent)
        # squared Mahalanobis distance from center to confidence ellipsoid
    max_displacments = np.sqrt(np.diag(covariance*conf_mdist_sq))
        # this gives the projection of the percent confidence ellipsoid
        # onto each coordinate direction - see notes for derivation
    return max_displacments