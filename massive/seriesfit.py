"""
Module for fitting data a model which is a sum of identical sub-models.
"""


import numpy as np
import scipy.optimize as opt


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
    set iteratively to be greater than the widths of the best-fitting
    sub-models. If two or more features are overlapping, they will be
    included in one sub-region and fit to a sum of multiple sub-models.
    """
    def __init__(self, x, y, submodel, get_width, get_center, initial_guess):
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
        get_center - func
            A function to determine the central value, in units of the
            dependent variable, of a single instance of a submodel.
            The calling syntax must be center(params), with params
            being the parameters of the submodel in question.
        get_width - func
            Similar to 'get_center' above, but returning the width
            in units of the dependent variable of the submodel. The
            width returned by get_width is the minimum region size
            used to fit a single sub-feature.
        initial_guess - arraylike
            An array of initial guesses for the parameters of each
            submodel, i.e. initial_guess[n] is an array of initial
            parameter guesses for submodel n. The length of the first
            axis determines the number of submodels to be used.
        """
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.submodel = submodel
        self.get_width = get_width
        self.get_center = get_center
        self.initial_guess = np.asarray(initial_guess)
        self.num_features = self.initial_guess.shape[0]
        self.num_subparams = self.initial_guess.shape[1]
        self.current_params = self.initial_guess.copy()

    def contained(self, i1, i2):
        """
        Is i1 contained in i2?
        """
        contained = (i1[0] >= i2[0]) and (i1[1] <= i2[1])
        return contained


    def partition(self):
        """
        Divide the model domain into the maximum possible number of
        independent sub-regions. In each such region, only a subset of
        the sub-models contributes noticeably to the full model.
        """
        centers = np.array([self.get_center(p) for p in self.current_params])
        widths =  np.array([self.get_width(p) for p in self.current_params])
        lowers = centers - 0.5*widths
        uppers = centers + 0.5*widths
        sequential = np.argsort(centers)
        first_feature = sequential[0]
        regions = [[lowers[first_feature], uppers[first_feature]]]
        features = [[first_feature]]
        for current in sequential[1:]:
            previous = current - 1
            overlap = (lowers[current] < uppers[previous])
            if overlap:
                regions[-1] = [lowers[previous], uppers[current]]
                features[-1].append(current)
            else:
                regions.append([lowers[current], uppers[current]])
                features.append([current])
        return features, np.asarray(regions)

    def get_residual_func(self, x, y, num_models):
        def resid(params):
            bkg = params[0]
            subparams = params[1:]
            restricted_model = bkg + sum(
                self.submodel(x, subparams[n*self.num_subparams:(n+1)*self.num_subparams])
                for n in xrange(num_models))
            return y - restricted_model
        return resid

    def run_fit(self):
        self.features, self.regions = self.partition()
        self.bkg = np.zeros(self.regions.shape[0])
        changed = True
        while changed:
            self.features, self.regions = self.partition()
            for region_index, ((xmin, xmax), models) in enumerate(zip(self.regions, self.features)):
                in_fit = ((xmin < self.x) & (self.x < xmax))
                target_x, target_y = self.x[in_fit], self.y[in_fit]
                num_models = len(models)
                resid = self.get_residual_func(target_x, target_y, num_models)
                starting_params = np.concatenate(([self.bkg[region_index]],
                    self.current_params[models, :].flatten()))
                fit_results = opt.leastsq(resid, starting_params)
                bestfit_bkg = fit_results[0][0]
                bestfit_params = fit_results[0][1:]
                self.bkg[region_index] = bestfit_bkg
                for submodel_index, model in enumerate(models):
                    self.current_params[model, :] = (
                        bestfit_params[submodel_index*self.num_subparams:(submodel_index + 1)*self.num_subparams])
            new_features, new_regions = self.partition()
            changed = new_regions.shape != self.regions.shape
            if not changed:
                changed = np.any([~self.contained(rnew, rold) for rold, rnew
                                  in zip(self.regions, new_regions)])
