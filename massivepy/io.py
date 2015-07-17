"""
Basic repetiitve io tasks
"""

import os
import re

import numpy as np
import pandas as pd

import utilities as utl
import massivepy.constants as const

def parse_paramfile_path(path):
    """
    Returns the directory and galaxy name, given a parameter file path.
    Checks that the first 7 characters of the file name match 'NGC####'.
    Also checks that the directory has #### somewhere in it.
    """
    output_dir = os.path.dirname(path)
    gal_name = os.path.basename(path)[0:7]
    if re.match(utl.force_full_match(const.re_ngc),gal_name) is None:
        raise Exception("Invalid galaxy name in parameter file path.")
    re_gal = re.search(const.re_ngc,output_dir)
    if re_gal is None:
        print "\nWarning, your output directory has no ngc name."
        print "Your organization is bad, go fix it.\n"
    elif re_gal.groups()[0] != gal_name[-4:]: #Test that galaxy number matches
        print "\nWarning, your output directory does not match your galaxy."
        print "Putting {} in {} directory\n".format(gal_name,re_gal.groups()[0])
    else:
        pass
    return output_dir, gal_name

def get_gal_center_pa(targets_path,gal_name):
    """
    Returns galaxy center and pa from Jenny's target file.
    Should add to this some galaxy name regex checking stuff.
    """
    target_positions = pd.read_csv(targets_path,
                                   comment='#', sep="[ \t]+",
                                   engine='python')
    gal_position = target_positions[target_positions.Name == gal_name]
    gal_center = gal_position.Ra.iat[0], gal_position.Dec.iat[0]
    gal_pa = gal_position.PA_best.iat[0]
        # .ita[0] extracts scalar value from a 1-element dataframe
    return gal_center, gal_pa

def get_friendly_ppxf_output(path):
    """
    Returns a friendly dict- and recarray-based set of ppxf output.
    Only return the data we actually use in plots and want in text files.
    """
    #Anything not used here should probably get dropped from the fits files
    # as well, except in a debug=True type case.
    #Will want to make sure none of this information could come from earlier
    # files like the binned spectra files, because we should avoid too much
    # duplication. E.g. should avoid getting the spectra from here.
    data, headers = utl.fits_quickread(path)
    friendly_data = {}

    # populate basic scalar parameters
    nbins = headers[0]['NAXIS2']
    nmoments = headers[0]['NAXIS1']
    npixels = headers[2]['NAXIS1']
    friendly_data['nmoments'] = nmoments
    friendly_data['nbins'] = nbins
    #friendly_data['npixels'] = npixels
    friendly_data['add_deg'] = headers[0]['ADD_DEG']
    friendly_data['mul_deg'] = headers[0]['MUL_DEG']

    # populate moment stuff
    dt = {'names':['moment','err','scalederr'],'formats':3*[np.float64]}
    friendly_data['gh'] = np.zeros((nbins,nmoments),dtype=dt)
    for ibin in range(nbins):
        for imom in range(nmoments):
            friendly_data['gh'][ibin,imom] = tuple(data[0][:,ibin,imom])

    # populate template stuff
    ntemps = headers[1]['NAXIS1']
    dt = {'names':['id','weight','flux','fluxweight'],
          'formats':[int]+3*[np.float64]}
    friendly_data['temps'] = np.zeros((nbins,ntemps),dtype=dt)
    for ibin in range(nbins):
        ii = np.argsort(data[1][1,ibin,:]) # sort by weight
        for i,field in enumerate(dt['names']): # need fields in fits file order
            friendly_data['temps'][field][ibin,:] = data[1][i,ibin,:][ii][::-1]
    if nbins==1:
        friendly_data['temps'] = friendly_data['temps'][0,:]
        ii = np.nonzero(friendly_data['temps']['weight'])
        friendly_data['temps'] = friendly_data['temps'][ii]

    # populate bin stuff
    dt = {'names':['id','chisq'],'formats':[int,np.float64]}
    friendly_data['bins'] = np.zeros((nbins,),dtype=dt)
    for ibin in range(nbins):
        friendly_data['bins'][ibin] = tuple(data[3][:2,ibin])

    # populate spectrum stuff
    # pretty sure this can go away since its all in the bin output!
    dt = {'names':['bestmodel'], 'formats':['<f8']}
    friendly_data['spec'] = np.zeros((nbins,npixels),dtype=dt)
    friendly_data['spec']['bestmodel'] = data[2][0, ...]

    # populate waves 
    # pretty sure this can go away since its all in the bin output!
    friendly_data['waves'] = data[6]

    return friendly_data


def get_friendly_ppxf_output_mc(path):
    """
    Returns a friendly dict- and recarray-based set of ppxf mc output.
    Only return the data we actually use in plots and want in text files.
    """
    data, headers = utl.fits_quickread(path)
    friendly_data = {}

    nmoments = headers[0]['NAXIS1']
    nruns = headers[0]['NAXIS2']
    nbins = headers[0]['NAXIS3']
    nthings = headers[0]['NAXIS4']

    friendly_data['nruns'] = nruns

    friendly_data['moments'] = np.zeros((nbins,nmoments,nruns))
    for ibin in range(nbins):
        for imom in range(nmoments):
            friendly_data['moments'][ibin,imom,:] = data[0][0,ibin,:,imom]

    friendly_data['err'] = np.zeros((nbins,nmoments))
    for ibin in range(nbins):
        for imom in range(nmoments):
            friendly_data['err'][ibin,imom] = np.std(data[0][0,ibin,:,imom])

    return friendly_data
