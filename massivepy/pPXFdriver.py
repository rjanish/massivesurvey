"""
This is a wrapper for ppxf, facilitating kinematic fitting.
"""


import warnings
import functools
import copy

import numpy as np
import ppxf

import utilities as utl
import massivepy.constants as const
import massivepy.spectrum as spec
import massivepy.templates as temp


class pPXFDriver(object):
    """
    This class is a driver for sets of pPXF fits, aiding in particular
    the preparation of spectra and the recording of results.

    Before allowing a pPXF fit to occur, this class will ensure that
    both the observed spectra and template spectra are log-sampled in
    wavelength and have identical spectral resolutions.

    All of the data and settings input to the fit are stored as class
    attributes. Intermediate computations and fit results are
    similarly stored. A pickle of the entire class will serve as an
    exact record of the fit, and I/O methods are provided to simplify
    recording of individual results.
    """
    PPXF_REQUIRED_INPUTS = {"add_deg": ["degree", int],
                            "mul_deg": ["mdegree", int],
                            "num_moments": ["moments", int],
                            "bias": ["bias", float]}
        # These are input parameter names and types needed by pPXF's caller

    def __init__(self, spectra=None, templates=None,
                 fit_range=None, initial_gh=None, **ppxf_kwargs):
        """
        Args:
        """
        self.spectra = copy.deepcopy(spectra)
        self.templates = copy.deepcopy(templates)
        self.fit_range = np.asarray(fit_range, dtype=float)
        if self.fit_range.shape != (2,):
            raise ValueError("Invalid fit range shape {}"
                             "".format(self.fit_range.shape))
        for setting in self.PPXF_REQUIRED_INPUTS:
            if setting not in ppxf_kwargs:
                raise ValueError("{} must be specified to fit spectra"
                                 "".format(setting))
        self.ppxf_kwargs = {}
        for setting, value in ppxf_kwargs.iteritems():
            ppxf_name, conversion = self.PPXF_REQUIRED_INPUTS[setting]
            self.ppxf_kwargs[ppxf_name] = conversion(value)
        self.initial_gh = np.asarray(initial_gh, dtype=float)
        if self.initial_gh.shape != (self.ppxf_kwargs["moments"],):
            raise ValueError("Invalid starting gh parameters shape {}, "
                             "must match number of moments to fit {}"
                             "".format(self.initial_gh.shape,
                                       self.ppxf_kwargs["moments"]))
        self.results = None
        # prepare copied spectra
        try:
            self.spectra.log_resample()
        except ValueError, msg:
            print "skipping spectra log_resample ({})".format(msg)
        self.to_fit = utl.in_linear_interval(self.spectra.waves,
                                             self.fit_range)
        self.target_flux = self.fit_range[1] - self.fit_range[0]
            # normalizing the flux to equal the numerical wavelength
            # range sets the numerical spectral values near 1
        self.get_flux = functools.partial(spec.SpectrumSet.compute_flux,
                                          interval=self.fit_range)
            # this computes the flux of a SpectrumSet only within the
            # fitting range - useful for normalization of templates
        self.get_flux.__name__ = "compute_flux_of_fitting_region"
        self.spectra = (
            self.spectra.get_normalized(
                norm_func=self.get_flux, norm_value=self.target_flux))

    def prepare_library(self, target_spec):
        """
        """
        # match spectral resolution
        spec_ir_test = target_spec.test_ir[0]
        spec_ir_inerp_func = utl.interp1d_constextrap(spec_ir_test[:, 0],
                                                      spec_ir_test[:, 1])
        # spec_waves = target_spec.waves
        # spec_ir = target_spec.metaspectra["ir"][0]
        # spec_ir_inerp_func = utl.interp1d_constextrap(spec_waves, spec_ir)
            # use to get ir(template_waves); this will require some
            # extrapolation beyond the spectra wavelength range, which
            # the above function will do as a constant
        template_waves = self.templates.spectrumset.waves
        single_ir_to_match = spec_ir_inerp_func(template_waves)
        num_temps = self.templates.spectrumset.num_spectra
        ir_to_match = np.asarray([single_ir_to_match,]*num_temps)
        matched_library = self.templates.match_resolution(ir_to_match)
        # resample
        logscale = target_spec.get_logscale()
        matched_library.spectrumset.log_resample(logscale)
        # norm
        matched_library.spectrumset = (
            matched_library.spectrumset.get_normalized(
                norm_func=self.get_flux, norm_value=self.target_flux))
        return matched_library

    def get_old_library(self, target_spec):
        bin_number = target_spec.ids[0]
        base = ("bin-spectra-check-20150522/results/ngc1600/results-updated"
                "/ngc1600mitchell_updated-bin{:02d}-3900.0_5300.0"
                "".format(int(bin_number)))
        template_path = "{}-templates.txt".format(base)
        templates = np.loadtxt(template_path).T
        tempwave_path = "{}-templates_waves.txt".format(base)
        templates_waves = np.loadtxt(tempwave_path)
        return temp.TemplateLibrary(
            spectra=templates, bad_data=np.zeros(templates.shape, dtype=bool),
            noise=np.zeros(templates.shape),
            ir=4.5*np.ones(templates.shape), # this is actually matched to Mitchell data
            spectra_ids=np.arange(templates.shape[0]),
            wavelengths=templates_waves,
            spectra_unit=const.flux_per_angstrom,
            wavelength_unit=const.angstrom,
            name='old-library', test_ir=None,
            catalog=np.arange(templates.shape[0]))

    def run_fit(self):
        """
        """
        output = []
        for spec_iter, target_id in enumerate(self.spectra.ids):
            target_spec = self.spectra.get_subset([target_id])
            # target_spec = target_spec.crop(self.fit_range) # testing
            exact_fit_range = utl.min_max(target_spec.waves)
            # matched_library = self.prepare_library(target_spec)
            matched_library = self.get_old_library(target_spec)
            template_range = utl.min_max(matched_library.spectrumset.waves)
            log_temp_start = np.log(template_range[0])
            print "temp", log_temp_start, template_range[0]
            log_spec_start = np.log(exact_fit_range[0])
            print "spec", log_spec_start, exact_fit_range[0]
            velocity_offset = (log_temp_start - log_spec_start)*const.c_kms
            logscale = target_spec.get_logscale()
            velscale = logscale*const.c_kms
            good_pix_mask = ~target_spec.metaspectra["bad_data"][0]
            good_pix_indices = np.where(good_pix_mask)[0]
                # np.where outputs tuple for some reason: (w,), with w being
                # an array containing integer indices where the input is True
            library_spectra_cols = matched_library.spectrumset.spectra.T
                # pPXF requires library spectra in columns of input array
            matched_library.spectrumset.write_to_fits("results-ppxf/temps_oldlibcopies-{}.fits".format(target_id))
            target_spec.write_to_fits("results-ppxf/specs_oldlibs-{}.fits".format(target_id))
            print self.ppxf_kwargs
            print "velscale", velscale
            print "vsyst", velocity_offset
            raw_output = ppxf.ppxf(library_spectra_cols,
                                   target_spec.spectra[0],
                                   target_spec.metaspectra["noise"][0],
                                   velscale, self.initial_gh,
                                   goodpixels=good_pix_indices,
                                   vsyst=velocity_offset, plot=False,
                                   quiet=True, **self.ppxf_kwargs)
            output.append(raw_output.sol)
        return np.asarray(output), self.spectra.ids.copy()

