

class SeriesFit(object):
    """
    Class for computing fits to models that are sums of some localized
    sub-model, such as a series of Gaussian peaks centered at various
    different locations.
    """
    def __init__(self, x, y, submodel, width, center,
                 initial_guess, width_criteria, minimizer):
        """
        Args:
        x - arraylike
            Dependent variable
        y - arraylike
            Observed independent variable
        submodel - func
            The localized model, of which a sum of multiple instances
            comprises the full model.  The call syntax must be
            submodel(arg, params), where arg is the dependent variable
            and params is an array of parameters.
        center - func
            A function to determine the central value, in units of the
            dependent variable, of a single instance of a submodel.
            The call syntax must be center(params), with params being
            the parameters of the submodel in question.
        width - func
            Similar to 'center' above, but returning the width in
            units of the dependent variable of the submodel.
        initial_guess - arraylike
            An array of initial guesses for the parameters of each
            submodel, i.e. initial_guess[n] is the set of initial
            parameter guesses for submodel n.  The length of the first
            axis determines the number of submodels to be used.
        width_criteria - float
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





