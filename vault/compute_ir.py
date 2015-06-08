
import pickle
import sys

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy.optimize import leastsq, fmin_powell
from numpy.polynomial.legendre import legval


target_filename = sys.argv[1]
reduced_dir = sys.argv[2]
ir_pickle_filename = sys.argv[3]

# read data
target = fits.open("{}/{}".format(reduced_dir, target_filename))
wavelengths = target[2].data[0, :]  # all rows identical, use first
redshift = target[4].header['Z']
wavelengths = wavelengths*(1 + redshift)  # stored in gal rest frame
arc_frame_raw = target[4].data
spectra = target[0].data
noise = target[1].data
target.close()
spectra_mask = ((np.absolute(spectra) > 10**4) |
                (np.absolute(noise) > 10**4))
bad_fibers = np.all(spectra_mask, axis=1)
flag = -666
rel_tol = 10**(-10)
masked = np.absolute(arc_frame_raw - flag) < rel_tol*arc_frame_raw
rescale = np.median(arc_frame_raw, axis=1)
arc_frame = ((arc_frame_raw.T)/rescale).T
arc_frame[masked] = -666
arc_frame[spectra_mask] = -666


def sum_of_gaussians(x, params):
    num_gaussian = int(params.shape[0]/3)
    heights = params[:num_gaussian]
    centers = params[num_gaussian:2*num_gaussian]
    sigmas = params[2*num_gaussian:]
    total = np.zeros(x.shape)
    for height, center, sigma in zip(heights, centers, sigmas):
        arg = (x - center)/sigma
        total += height*np.exp(-0.5*arg**2)
    return total


def arc_model(wavelengths, params):
    wmax, wmin = np.max(wavelengths), np.min(wavelengths)
    x = (2*wavelengths - wmin - wmax)/(wmax - wmin)
    offset = legval(x, params[:1])
    gaussians_params = params[1:]
    return offset + sum_of_gaussians(wavelengths, gaussians_params)
    

centers = np.array([4046.5469, 4077.8403, 4358.3262, 4678.1558, 4799.9038,
                    4916.0962, 5085.8110, 5460.7397, 5769.5972])
# sigmas = np.ones(centers.shape)*4.8/2.35
sigmas = np.ones(centers.shape)*4.5/2.35
heights = np.ones(centers.shape)*np.max(arc_frame)
offset = np.array([np.median(arc_frame)])
initial_params = np.concatenate((offset, heights, centers, sigmas))

cropped_edges = (3800 < wavelengths) & (wavelengths < 5780)

bf_sigmas = []
bf_centers = []
bf_models = []
for n, fiber in enumerate(arc_frame):
    # masking
    valid = ((0.0 < fiber) & cropped_edges)
    num_datapts = np.sum(valid)
    if num_datapts < 100:  # bad fiber
        current_centers = centers.copy()
        current_sigmas = np.ones(centers.shape)*np.nan
        current_model = np.ones(wavelengths[valid].shape)*np.nan
    else:
        # fit
        print 'running fit'
        objective = lambda p: np.sum((fiber[valid] - arc_model(wavelengths[valid], p))**2)
        fit = fmin_powell(objective, initial_params, full_output=True)
        # store results
        bestfit = fit[0]
        bf_offset = bestfit[:1]
        gauss_bestfit = bestfit[1:]
        divider = int(gauss_bestfit.shape[0]/3)
        current_centers = gauss_bestfit[divider:2*divider]
        current_sigmas = gauss_bestfit[2*divider:]
        # initial_params = bestfit
        current_model = arc_model(wavelengths[valid], bestfit)
        # plt.plot(wavelengths, fiber, marker='o',
        #          linestyle='', color='k', alpha=0.3)
        # plt.plot(wavelengths[valid], fiber[valid], marker='o',
        #          linestyle='', color='k', alpha=0.8)
        # plt.plot(wavelengths[valid], current_model, marker='',
        #          linestyle='-', color='r', alpha=0.8)
        # plt.show()
    bf_models.append(current_model)
    bf_centers.append(current_centers)
    bf_sigmas.append(current_sigmas)
bf_sigmas = np.array(bf_sigmas)
bf_fwhms = bf_sigmas*(2*np.sqrt(2*np.log(2)))
bf_centers = np.array(bf_centers)
bf_models = np.array(bf_models)
# plot
fig, ax = plt.subplots()
plt.plot(bf_centers.flatten(), bf_fwhms.flatten(), 'bo', alpha=0.4)
plt.show()
# write output
results = np.array([bf_centers, bf_fwhms])
with open(ir_pickle_filename, 'wb') as pickled:
    pickle.dump(results, pickled)
np.savetxt('{}.txt'.format(ir_pickle_filename), results)