

class SeriesFit(object):
    """
    Class for computing fits to models that are sums of some localized
    sub-model, such as a series of Gaussian peaks centered at various
    different locations.  
    """
    def __init__(self, data, submodel, num_submodels, 
                 initial_guess, width_criteria, minimizer):
        pass



from astropy.io import fits
import numpy as np

from gausshermite import gausshermite_pdf 

testdata_filename = '/stg/current/ngc1600/data/mitchell/QnovallfibNGC1600_log.fits'
testdata_hdu = fits.open(testdata_filename)
testdata = testdata_hdu[4].data
testdata_hdu.close()
test_spectrum = testdata[0, :]/np.median(testdata[0, :])

