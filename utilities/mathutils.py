""" Miscellaneous mathematical functions """


import numpy as np
import scipy.interpolate as interp


def gaussian(x, mu, sigma):
    """
    Returns the normalized Gaussian G(x) evaluated element-wise on x.
    G(x) has a mean mu and standard deviation sigma, and is normalized
    to have a continuous integral from x=-inf to x=inf of 1.
    """
    x = np.asarray(x)
    mu, sigma = float(mu), float(sigma)
    arg = (x - mu)/sigma
    norm = sigma*np.sqrt(2*np.pi)
    return np.exp(-0.5*(arg**2))/norm


def cartesian_product(arrays, output_dtype=None):
    """
    Generate a Cartesian product of input arrays.

    Args:
    arrays - list
        List of arrays with which to form the Cartesian
        product.  Input arrays will be treated as 1D.
    output_dtype - numpy data type
        The dtype of the output array. If unspecified, will be
        taken to match the dtype of the first element of arrays.

    Returns:
    output - 2D array
        Array of shape (arrays[0].size*arrays[1].size..., len(arrays))
        containing the Cartesian product of the input arrays.

    Examples:
    >>> cartesian_product(([1, 2, 3, 4], [5.0], [6.2, 7.8]))
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


def in_linear_interval(array, interval):
    """
    Indicate the array elements that are contained in interval.

    Args:
    array - arraylike
        Array to test for containment, assumed to be numeric-valued.
    interval - 1D, 2-element arraylike
        A 1D interval expressed as [start, end], assumed start <= end.

    Returns:
    contained - boolean arraylike
        A boolean array of the same shape as the input array, valued
        True if the the input array values are in the passed interval.
    """
    array = np.asarray(array, dtype=float)
    start, end = np.asarray(interval, dtype=float)
    if start > end:
        raise ValueError("Invalid interval: ({}, {})".format(start, end))
    return (start < array) & (array < end)


def in_periodic_interval(array, interval, period):
    """
    Indicate the array elements that are contained in the passed
    interval, taking into account the given periodicity.

    For a passed interval = [a, b], the interval is defied as the
    region starting at a and continuing in the direction of explicitly
    increasing values to b, all subject to the passed identification
    x ~ x + period.  I.e., if a and b are interpreted as angles on the
    circle, the interval is the arc from a to b counterclockwise.
    There is no restriction a < b: the intervals [a, b] and [b, a] are
    both sensible, representing the two paths along the circle
    connecting points a and b.

    Args:
    array - arraylike
        Array to test for containment, assumed to be numeric-valued.
    interval - 1D, 2-element arraylike
        A 1D interval expressed as [start, end], see convention above.

    Returns:
    contained - boolean arraylike
        A boolean array of the same shape as the input array, valued
        True if the the input array values are in the passed interval.
    """
    period = float(period)
    array = np.asarray(array, dtype=float)
    start, end = np.asarray(interval, dtype=float)
    dist_to_data = (array - start) % period
    dist_to_interval_end = (end - start) % period
    return dist_to_data < dist_to_interval_end


def in_union_of_intervals(values, intervals, inclusion=in_linear_interval):
    """
    Are values contained in a union of inclusive 1D intervals?

    This function implements only the union logic. It can be used with
    any notion of point-in-interval via the passed inclusion function.

    Args:
    values - arraylike
        The values to be tested for inclusion.
    ranges - iterable
        An iterable of 1D inclusive intervals, each interval
        specified as a range [lower, upper].
    inclusion - func
        A function that determines if a value is in a given interval,
        accepting trial array as the first argument, an interval
        [lower, upper] as the second, and returning a boolean array.

    Returns: in_union
    in_union - bool array, shape matching values
        A boolean of the same shape as the input values, valued True
        if the input values are in the union of the passed intervals.
    """
    in_union = np.zeros(values.shape, dtype=np.bool)
    for interval in intervals:
        in_interval = inclusion(values, interval)
        in_union = in_union | in_interval
    return in_union


def interval_contains_interval(interval1, interval2,
                               inclusion=in_linear_interval):
    """
    Is interval2 contained in interval1? Boolean output.

    This function implements the notion of an interval contained in
    another interval using only an abstracted notion of
    point-in-interval containment: interval2 is contained in interval1
    if both of the endpoints of interval2 are contained in interval1.
    The point-in-interval notion is defined via the user-supplied
    function inclusion, allowing this function to be used with any
    sort of values - linear, periodic, etc. By default, it assumes
    containment in the usual sense of the real number line.

    Args:
    interval1 - 1D, 2-element arraylike
    interval2 - 1D, 2-element arraylike
        Will test if interval2 is contained in interval1.
        The intervals should have the form [start, end], with start
        and end possibly subject to some criteria depending on the
        containment function used. The default linear containment
        requires that start <= end.
    inclusion - func
        A function that determines if a value is in a given interval,
        accepting trial array as the first argument, an interval
        [lower, upper] as the second, and returning a boolean array.

    Returns: in_union
    in_union - bool value
        True if interval2 is contained in interval1.
    """
    interval1 = np.asarray(interval1, dtype=float)
    start2, end2 = np.asarray(interval2, dtype=float)
    return inclusion(start2, interval1) & inclusion(end2, interval1)


def min_max(array, axis=None):
    """ Return the minimum and maximum of passed array """
    array = np.asarray(array)
    return np.asarray([array.min(axis=axis), array.max(axis=axis)])


def quartiles(array, axis=None):
    """ Return the quartile boundaries of passed array """
    array = np.asarray(array)
    zero, one = min_max(array, axis=axis)
    quarter = np.percentile(array, 25, axis=axis)
    half = np.median(array, axis=axis)
    threequarters = np.percentile(array, 75, axis=axis)
    return np.asarray([zero, quarter, half, threequarters, one])


def clipped_mean(data, weights=None, noise=None,
                 mask=None, fill_value=None, clip=5, max_iters=5,
                 max_fractional_remove=0.02, converge_fraction=0.02):
    """
    Compute a clipped, weighted mean of each column in the passed 2d
    array.  This is the weighted mean excluding any data points that
    differ from the unweighted median by greater than the passed clip
    factor times the standard deviation. This is iterative.

    Args:
    data - 2d ndarray
        Array on which to compute the clipped, weighted column mean.
    weights - 1d or 2d ndarray
        The weighting factors to be used.  Will be normalized so that
        each column of weights sums to 1.  If a 1d array is passed,
        the same weights will be used for each column. If unspecified,
        then uniform weights are used.
    noise - 2d ndarray, defalt=None
        The estimated noise in the passed data, will be ignored if not
        passed or combined in quadrature if an array is given
    fill_value - float, default=None
        If given, will be used to fill all mask values in the final
        output, otherwise masked values are allowed to float.
    clip - float, default=3
        All data differing from the unweighted median by more than
        clip*standard_deviation will be ignored
    max_iters - int, default=10
        maximum number of iterations before clipping is halted
    max_fractional_remove - float, default=0.02
        maximum fraction number of data points
        that can be clipped before clipping is halted
    mask - 2d boolean ndarray
        True for any pixels to be ignored in the computation.

    Returns: clipped_data_mean, clipped_data_noise (optional),
             mean_mask, clipped
    clipped_data_mean - 1d ndarray
        The clipped, weighted mean of each column of data
    clipped_data_noise - 1d ndarray
        The noise estimate in the clipped mean of each column, only
        returned if a noise array is passed
    mean_mask - 1d ndarray
        The masking array for the mean data, indicated any columns
        for which all values were either masked or clipped
    clipped_points - 2d boolean ndarray
        An array of the same size as the input data, with a True for
        every data point that was clipped
    """
    data = np.asarray(data, dtype=float)
    if mask is None:
        masked = np.zeros(data.shape, dtype=bool)
    else:
        masked = np.asarray(mask, dtype=bool)
    total_num_points = float(data.size)  # float for fractional divisions
    if total_num_points == 0:
        raise ValueError("Data arrays must be non-empty.")
            # division by total number of points is needed below
    masked_data = np.ma.array(data, mask=masked)
    clipped = np.zeros(data.shape, dtype=bool)
    # determine values to clip
    for iter in xrange(max_iters):
        sigma = np.ma.std(masked_data, axis=0)
        central = np.ma.median(masked_data, axis=0)
        distance = np.ma.absolute(masked_data - central)/sigma
            # default broadcasting is to copy vector along each row
        new_clipped = (distance > clip).data
            # a non-masked array, any data already masked in distance are set
            # False by default in size compare - this finds new clipped only
        num_old_nonclipped = np.sum(~clipped)
        clipped = clipped | new_clipped  # all clipped points
        masked_data.mask = clipped | masked  # actual clipping
        total_frac_clipped = np.sum(clipped)/total_num_points
        delta_nonclipped = np.absolute(np.sum(~clipped) - num_old_nonclipped)
        delta_frac_nonclipped = delta_nonclipped/float(num_old_nonclipped)
        if ((delta_frac_nonclipped <= converge_fraction) or  # convergence
            (total_frac_clipped > max_fractional_remove)):
            break
    # compute mean
    bad_pixels = masked | clipped
    if weights is None:
        weights = np.ones(data.shape)
    else:
        weights = np.array(weights, dtype=float)
    if weights.ndim == 1:
        weights = np.vstack((weights,)*masked_data.shape[1]).T
            # array with each row a constant value
    weights[bad_pixels] = 0.0  # do not include clipped or masked in norm
    total_weight = weights.sum(axis=0)
    all_bad = np.all(bad_pixels, axis=0)
    total_weight[all_bad] = 1.0
        # set nonzero fiducial total weight for wavelengths with no un-masked
        # values to avoid division errors; normalized weight is still zero
    weights = weights/total_weight  # divides each col by const
    clipped_data_mean = np.ma.sum(masked_data*weights, axis=0).data
    mean_mask = np.all(bad_pixels, axis=0)
    if fill_value is not None:
        clipped_data_mean[mean_mask] == fill_value
    if noise is not None:
        noise = np.asarray(noise, dtype=float)
        masked_noise = np.ma.masked_array(noise, mask=bad_pixels)
        clipped_variance = np.ma.sum((masked_noise*weights)**2, axis=0).data
        clipped_data_noise = np.sqrt(clipped_variance)
        if fill_value is not None:
            clipped_data_noise[mean_mask] == fill_value
        return clipped_data_mean, clipped_data_noise, mean_mask, clipped
    else:
        return clipped_data_mean, mean_mask, clipped


def interp1d_constextrap(*args, **kwargs):
    """
    This is a wrapper for scipy.interpolate.interp1d that allows
    a constant-valued extrapolation on either side of the input data.

    The arguments are the same as interp1d, and the return is a
    function analogous to that returned by interp1d. The only
    difference is that upon extrapolation the returned function will
    not throw an exception, but rather extend the interpolant as a
    flat line with value given by the nearest data value.
    """
    interp_func = interp.interp1d(*args, **kwargs)
    x, y = args[:2]  # convention from interp1d
    xmin_index = np.argmin(x)
    xmin = x[xmin_index]
    ymin = y[xmin_index]
    xmax_index = np.argmax(x)
    xmax = x[xmax_index]
    ymax = y[xmax_index]
    def interp_wrapper(z):
        """ 1d interpolation with constant extrapolation """
        z = np.asarray(z, dtype=float)
        is_lower = z <= xmin
        is_upper = xmax <= z
        valid = (~is_lower) & (~is_upper)
        output = np.nan*np.ones(z.shape)
        output[is_lower] = ymin
        output[is_upper] = ymax
        output[valid] = interp_func(z[valid])
        return output
    return interp_wrapper





