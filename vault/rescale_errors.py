"""
Re-scale the error arrays of NGC1600 by the chisq of 6-param, full
spectral range fits to the selected12 templates.
"""


import numpy as np
from astropy.io import fits

from util import re_filesearch


spectra_source = 'binned-spectra1600-trim/'
souce_pattern = r'bin(?P<bin_number>\d{2})_ngc1600.fits'
spectra_dist = 'binned-spectra1600-rescalederror/'
results_dir = 'results1600-real12-6/'
bestfit_chisq_pattern = (r"ngc1600-4103.65_5684.44-chisq_dof\.txt")
bestfit_binnames_pattern = (r"ngc1600-4103.65_5684.44-bin_number\.txt")

chisq_file_matches = re_filesearch(bestfit_chisq_pattern, results_dir)
binname_file_matches = re_filesearch(bestfit_binnames_pattern, results_dir)
chisq_dofs = np.loadtxt(chisq_file_matches[0][0])
bin_names = np.loadtxt(binname_file_matches[0][0]).astype(int)

spectra_matches = re_filesearch(souce_pattern, spectra_source)
spectra_paths = {int(m.groups()[0]):f for f, m in zip(*spectra_matches)}

print 're-scaling spectra:'
for bin_iter, (bin_num, path) in enumerate(spectra_paths.iteritems()):
    chisq_dof = chisq_dofs[bin_iter]
    print 'bin {:>2s}:  chisq/dof = {:>5.2f}'.format(str(bin_num), chisq_dof)
    hdu = fits.open(path)
    bin_data = hdu[0].data
    bin_header = hdu[0].header
    hdu.close()
    bin_header.add_comment("error array re-scaled by pPXF fit chisq")
    bin_data[1, :] *= np.sqrt(chisq_dof)
    new_path = ("{}/bin{:02d}_ngc1600_rescalederror.fits"
                "".format(spectra_dist, bin_num))
    fits.writeto(new_path, bin_data, header=bin_header)



