
import os

import numpy as np
import matplotlib.pyplot as plt

import utilities as utl

spectrumplots = [3001, 3000]
errorplots = [3001, 3000]
plot_region = [8600, 8850]
norm_region = [8710, 8750]
data_dir = "data/gmos/binned"
results_dir = "gmos_ir_update"
params_filename = "ngc1600gmos-8440.0_8750.0-add0_mult3-gh_params.txt"
errors_filename = "ngc1600gmos-8440.0_8750.0-add0_mult3-gh_params_errors.txt"
binnames_filename = "ngc1600gmos-8440.0_8750.0-add0_mult3-bin_name.txt"

data = {}
fig, ax = plt.subplots()
for b in spectrumplots:
    name = 'vor{:04d}.dat'.format(b)
    path = os.path.join(data_dir, name)
    waves, spectrum, good = np.loadtxt(path).T
    valid = (utl.in_linear_interval(waves, plot_region) &
             np.asarray(good, dtype=bool))
    to_norm = (utl.in_linear_interval(waves, norm_region) &
               np.asarray(good, dtype=bool))
    normed = spectrum/np.median(spectrum[to_norm])
    data[b] = np.asarray([waves[valid], normed[valid]]).T
    ax.plot(waves[valid], normed[valid], alpha=0.6, label=name)
ax.legend(loc='best')
fig.savefig("ngc1600gmos_central_spectra.png")


params = np.loadtxt(os.path.join(results_dir, params_filename))
errors = np.loadtxt(os.path.join(results_dir, errors_filename))
binnames = np.loadtxt(os.path.join(results_dir, binnames_filename), dtype=str)
fig, ax = plt.subplots()
n_used = []
for n, (name, p, e) in enumerate(zip(binnames, params.T, errors.T)):
    if int(name[3:]) in errorplots:
        ax.errorbar(n, p[1], yerr=e[1], alpha=0.4, label=name,
                    marker='o', linestyle='')
        n_used.append(n)
ax.legend(loc='best')
ax.set_xlim([-10, np.max(n_used) + 10])
fig.savefig("ngc1600gmos_central_sigma.png")

plt.show()