""" 
Functions to compute Hermite polynomials and Gauss-Hermite distributions
"""


import numpy as np
from scipy.integrate import quad

from mathutils import gaussian


def hermite_polynomials(sample_pts, num_polynomials):
    """
    Return the first num_polynomials Hermite polynomials, i.e. all
    H_n, for 0 <= n <= (num_polynomials - 1), evaluated element-wise
    on sample_pts. Returns an array such that: output[n, ...] is
    H_n(sample_pts), which itself has the same shape as sample_pts.
    These are computed recursively from the first two polynomials.

    Gauss-Hermite definition is:
        $ H_n(x) = \left( -1 \right)^n \frac{1.0}{\sqrt{n! \, 2^n}}
                    e^{x^2} \left( \frac{d^n}{dx^n} \right) e^{-x^2} $
    this matches van der Marel & Franx (1993) and pPXF, but does not
    match either of the two definitions in Abramowitz & Stegun

    This gives the first two polynomials as above are:
    $ H_0(x) = 1 $
    $ H_1(x) = \sqrt{2} x $
    and subsequent polynomials determined by the recursion:
    $ H_{n}(x) = \frac{1}{\sqrt{n}} \left[ \sqrt{2} x H_{n-1}(x)
                        - \sqrt{n-1} H_{n-2}(x) \right] $
    """
    sample_pts = np.asarray(sample_pts)
    sample_pts_shape = sample_pts.shape
    num_polynomials = int(num_polynomials)
    output_shape = [num_polynomials] + list(sample_pts_shape)
    hermites = np.zeros(output_shape)
    hermites[0, ...] = np.ones(sample_pts_shape)  # H_0(x) = 1
    hermites[1, ...] = np.sqrt(2)*sample_pts      # H_1(x) = sqrt(2)*x
    for n in xrange(2, num_polynomials):  # loop (num_polynomials - 2) times
        hermites[n, ...] = (np.sqrt(2)*sample_pts*hermites[n - 1, ...] -
                          np.sqrt(n - 1)*hermites[n - 2, ...])/np.sqrt(n)
    return hermites


def hermite_series(x, params):
    """
    Compute the Hermite series: $\sum_i params[i] H_i(x)$, where $H_i$
    are the Hermite polynomials as defined in hermite_polynomials.

    Args:
    x - array
        Argument values on which to evaluate the series, can be any
        shape and the series will be evaluated element-wise on x.
    params - 1d array
        Coefficients of each Hermite polynomial in the series,
        assumed to start with the overall additive constant, i.e.
        the coefficient of $H_0(x) = 1$, and end with the coefficient
        of $H_{len(params) - 1}(x)$.

    Returns: total
    total - 1d array
        The sum $\sum_i params[i] H_i(x)$, given as an array for each
        input values of x.  Has the same shape as the input x.
    """
    x, params = np.asarray(x), np.asarray(params)
    num_params = params.shape[0]
    polynomials = hermite_polynomials(x, num_params)
    total = np.zeros(x.shape)
    for index, param in enumerate(params):
        total += param*polynomials[index, ...]
    return total


def unnormalized_gausshermite_pdf(x, params):
    """
    Returns a *un*-normalized Gauss-Hermite probability density
    function with the passed parameters, evaluated element-wise on x.

    This is identical to the function gausshermite_pdf, except that
    the normalization factor N in the definition of the Gauss-Hermite
    distribution is set to 1 regardless of the input parameters.

    For math conventions and usage syntax, see gausshermite_pdf.
    """
    x = np.asarray(x)
    params = np.asarray(params)
    central, sigma = params[:2]
    gaussian_factor = gaussian(x, central, sigma)  # Gaussian is normalized
    full_h_series = np.concatenate(([1.0, 0.0, 0.0], params[2:]))
    scaled_input = (x - central)/sigma
    poly_factor = hermite_series(scaled_input, full_h_series)
    return gaussian_factor*poly_factor # full product is unnormalized


def gausshermite_pdf(x, params):
    """
    Returns a normalized Gauss-Hermite probability density function
    with the passed parameters GH(x) evaluated element-wise on x.

    The Gauss-Hermite distribution used here is:
    $ GH(x; mu, sigma, {h_i}) =
        N G(x; mu, sigma) (1 + \sum_{i=3} h_i H_i[(x - mu)/sigma]) $
    where the GH parameters are v, sigma, and $h_i$ for $i > 2$. G is
    the Gaussian pdf with mean mu and standard deviation sigma,
    normalized to integrate to 1 as usual. The $H_i$ are the Hermite
    polynomials as defined in hermite_polynomials. N is an overall
    normalization constant which forces the GH distribution to
    integrate to 1.

    Args:
    x - arraylike
        Argument on which GH is evaluated, can have any shape.
    params - 1d array
        GH parameters of the distribution:
        params[0] - central value, called 'mu' above,
                    must have the same units as x
        params[1] - width, called 'sigma' above,
                    must have the same units as x
        params[j], j > 1 -  the unitless Hermite parameter h_{j+1}.
        
        The first two elements of params are required while the rest
        are optional, with any h_i not specified assumed to be zero.
        The h_n values given in params[2:] are reckoned in order to be
        h_3, h_4, h_5, etc, and so if some nonzero h_m is to be
        included, params must have length of at least m.

    Returns: gh_x
    gh_x - array
        GH(x) with the passed Gauss-Hermite parameters evaluated
        element-wise on x. Has the same shape as x.
    """
    func = lambda y: unnormalized_gausshermite_pdf(y, params)
    norm, error = quad(func, -np.inf, np.inf)
    return func(x)/norm


def fixparam_gausshermite_pdf(params, normalize=True):
    """
    Returns a callable Gauss-Hermite probability density function
    with the Gauss-Hermite parameters fixed to those passed here.

    This function is equivalent to a lambda call that freezes the 
    values of the Gauss-Hermite parameters in gausshermite_pdf. I.e.,
    >>> f1 = fixparam_gausshermite_pdf(params)
    >>> f2 = lambda x: gausshermite_pdf(x, params)
    then f1(x) == f2(x) is True for any x.  

    This function optimizes run-time in the case that one must make
    multiple calls to a GH pdf. If lambda is used as above, then the
    GH distribution is fully integrated with each call to f2 in order
    to set the normalization. In contrast, the current function sets
    the normalization only once while constructing the output function,
    so f1 does not need to do an integration. f1 is therefore faster
    than f2, by about the time needed for one full integration of the
    GH pdf.  Note, however, that since the current function must do
    one integration, the total time needed for the two calls
    >>> f1 = fixparam_gausshermite_pdf(p)
    >>> values = f1(x)
    will be roughly the same as the time needed for the one call 
    >>> values = gausshermite_pdf(x, p)
    for any x and p. The speed increase is useful only if f1 is to be
    called multiple times.

    For math conventions see gausshermite_pdf.

    Args:
    params - 1d array
        GH parameters of the distribution to be fixed.
        For syntax see gausshermite_pdf.

    Returns: gh
    gh - function
        A function GH, for which GH(x) evaluates the GH distribution
        with the above passed parameters. Evaluation is element-wise
        on x, so that the output has the same shape as x. The units
        of x must be the same as the units of the central value
        params[0] and width params[1] passed above.
    """
    params = np.asarray(params)
    unnormed_pdf = lambda y: unnormalized_gausshermite_pdf(y, params)
    # construct docstring for returned func
    doctxt = ("Returns {} Gauss-Hermite pdf GH(x) evaluated"
              " element-wise on x. GH(x) has the parameters:\n"
              "".format('a normalized' if normalize else 'an unnormalized'))
    gausstxt = "central: {}\nwidth:   {}\n".format(*params[:2])
    hermitetxt = '\n'.join(["h{}:      {}".format(n + 3, p)
                            for n, p in enumerate(params[2:])]) + '\n'
    zerostxt = "h_i:     0.0, for i > {}".format(params.size)
    full_docstring = doctxt + gausstxt + hermitetxt + zerostxt
    if normalize:
        norm, error = quad(unnormed_pdf, -np.inf, np.inf)
        normed_pdf = lambda x: unnormed_pdf(x)/norm
        normed_pdf.func_doc = full_docstring
        return normed_pdf
    else:
        unnormed_pdf.func_doc = full_docstring
        return unnormed_pdf



