"""
Module for fitting data a model which is a sum of identical sub-models.
"""

class SeriesFit(object):
    """
    This class simplifies fitting data to models which consist of a
    sum of multiple localized and identically-shaped sub-models.

    The model $F$ should take the form:
      $ F(x; P) = b(x; p_b) + \sum_{i=0}^{N-1} f(x; p_i) $.
    Here x is the data. b is a slowly-varying background model with a
    set of $n_b$ parameters denoted collectively by $p_b$. Similarly,
    f is a localized model (such as a Gaussian peak) dependent on a
    set of $n_p$ parameters. There are $N$ such sub-models, each with
    an independent parameter set denoted $p_i$ for the ith sub-model.
    This parameters set $p_i$ must determine $f$'s size and central
    location. The data will be fit to determine the $N n_p + n_b$
    parameters of the full model.

    The sub-model $f$ is assumed to describe the interesting data
    features, with $b$ an uninteresting background. Fitting to $b$
    is avoided as much as possible to improve performance. This is
    done by partitioning the data into sub-regions and fitting one $f$
    per region independent of all other regions. The region sizes are
    set iteratively using the widths of the best-fitting sub-models.
    If two or more features are overlapping, they will be included
    in one sub-region and fit to a sum of multiple sub-models.
    """
    def __init__(self, x, y, submodel, width, center,
                 initial_guess, minimizer, width_criteria=5.0):
        """
        Args:
        x - arraylike
            Dependent variable
        y - arraylike
            Observed independent variable
        submodel - func
            The localized sub-model. The calling syntax must be
            submodel(arg, params), where arg is a dependent variable
            array and params is an array of parameters.
        center - func
            A function to determine the central value, in units of the
            dependent variable, of a single instance of a submodel.
            The calling syntax must be center(params), with params
            being the parameters of the submodel in question.
        width - func
            Similar to 'center' above, but returning the width in
            units of the dependent variable of the submodel.
        initial_guess - arraylike
            An array of initial guesses for the parameters of each
            submodel, i.e. initial_guess[n] is an array of initial
            parameter guesses for submodel n. The length of the first
            axis determines the number of submodels to be used.
        width_criteria - float, default=5.0
            This determines the size of the fitting region used for
            each submodel. Each region has size width_criteria*width,
            where width is the width of the submodel in that region.
        minimizer - func
            This is the routine used to minimize chi^2 of the model.
        """
        pass

    def full_model(x, params):
        split_params = [params[index:(index + self.num_subparams)]
                        for index in xrange(self.num_submodels)]
        return sum(submodel(x, subparams) for subparams in split_params)





