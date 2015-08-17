"""
Basic repetiitve io tasks
"""

import os
import re
import time
import functools

import numpy as np
import pandas as pd

import utilities as utl
import massivepy.constants as const

# regex patterns for extracting galaxy number from file paths
re_gals = {'NGC': r"(?:NGC|ngc|N|n)(?P<num>\d{4})",
           'UGC': r"(?:UGC|ugc|U|u)(?P<num>\d{5})"}

# regex patterns for extracting galaxy name/number from parameter file path
# this one is stricter, because the resulting galaxy name must fit the format
# of Jenny's target positions file.
re_gals_strict = {'NGC': r"NGC(?P<num>\d{4})(?=\D|$)",
                  'UGC': r"UGC(?P<num>\d{5})(?=\D|$)"}


def parse_paramfile_path(path):
    """
    Returns the directory and galaxy name, given a parameter file path.

    If the parameter file path does not contain a valid galaxy name,
    returns 'unknown' as the galaxy name. This should then trip pathcheck()
    to return False when it is called on the galaxy name, so in that case
    the galaxy would be skipped.

    This does not check for the validity of parameter file paths, because
    I always use tab completion to put them in, but that would be easy
    enough to add.
    """
    output_dir = os.path.dirname(path)
    filename = os.path.basename(path)
    gal_name = 'unknown'
    for re_type in re_gals_strict:
        gal_match = re.match(re_gals_strict[re_type],filename)
        if gal_match is None:
            continue
        gal_type = re_type
        gal_num = gal_match.group('num')
        gal_name = gal_match.group(0)
        break
    return output_dir, gal_name

def pathcheck(paths,extensions,gal_name):
    """
    Check that all paths meet the following criteria:
    -They exist.
    -They have the correct extension. Extensions should include the period
     (e.g. '.txt') for files or be empty (e.g. '') for dirs.
    -They are for the correct galaxy, based on regex patterns. Note that
     gal_name is parsed again for gal_type and gal_num, so that if we make
     the strict parsing of the parameter file name more flexible in the
     future (e.g. allowing ngc instead of NGC) this should not break.

    Failing either of the first two will return False, which should be
    used in the main scripts to continue on with the next galaxy.
    Failing the third will just print a warning message, but return True
    to indicate the path validation was successful.
    """
    gal_type = 'unknown'
    for re_type in re_gals_strict:
        gal_match = re.match(re_gals_strict[re_type],gal_name)
        if gal_match is None:
            continue
        gal_type = re_type
        gal_num = gal_match.group('num')
        break
    if gal_type == 'unknown':
        print 'Something went wrong extracting galaxy name {}'.format(gal_name)
        return False
    wrong_types = re_gals.keys()
    wrong_types.remove(gal_type)
    for path, ext in zip(paths,extensions):
        if not os.path.exists(path):
            print 'Path does not exist: {}'.format(path)
            return False
        if not os.path.splitext(path)[1]==ext:
            print 'Path has wrong extension, needs {}: {}'.format(ext,path)
            return False
        gal_matches = re.findall(re_gals[gal_type],path)
        gal_badmatches = []
        for wrong_type in wrong_types:
            gal_badmatches.extend(re.findall(re_gals[wrong_type],path))
        if gal_matches is None:
            print "\nWarning, your output directory has no ngc name."
            print "Your organization is bad, go fix it.\n"
        elif not all([gal_num==num for num in gal_matches]):
            print '----------------------------------------------------'
            print 'WARNING! WARNING! YOU SEEM TO BE COMBINING GALAXIES!'
            print 'THE FOLLOWING INPUT PATH IS A PROBLEM:'
            print path
            print 'BECAUSE YOU CLAIM TO BE DOING THIS GALAXY:'
            print gal_name
            print 'THE CODE WILL STILL RUN, BUT YOUR STUFF MAY BE WRONG!'
            print '----------------------------------------------------'
        elif not len(gal_badmatches)==0:
            print '----------------------------------------------------'
            print 'THE FOLLOWING INPUT PATH IS A PROBLEM:'
            print path
            print 'BECAUSE YOU CLAIM TO BE DOING THIS GALAXY:'
            print gal_name
            print 'THE CODE WILL STILL RUN, BUT YOUR STUFF MAY BE WRONG!'
            print '----------------------------------------------------'
        else:
            pass
    return True

def get_gal_center_pa(targets_path,gal_name):
    """
    Returns galaxy center and pa from Jenny's target file.
    File name should match one of the choices in the if/elif block below.
    """
    target_positions = pd.read_csv(targets_path,
                                   comment='#', sep="[ \t]+",
                                   engine='python')
    gal_position = target_positions[target_positions.Name == gal_name]
    if os.path.basename(targets_path)=='target-positions-pre20150731.txt':
        print 'Using old targets file:\n  {}'.format(targets_path)
        gal_center = gal_position.Ra.iat[0], gal_position.Dec.iat[0]
        gal_pa = gal_position.PA_best.iat[0]
        gal_re = np.nan
    elif os.path.basename(targets_path)=='target-positions.txt':
        gal_center = gal_position.RA.iat[0], gal_position.Dec.iat[0]
        gal_pa = gal_position.PA_NSA.iat[0]
        gal_re = gal_position.Re_NSA.iat[0]
        if gal_pa==-99.0:
            print 'NSA PA not available, using 2MASS'
            gal_pa = gal_position.PA_2MASS.iat[0]
        if gal_pa < 0:
            gal_pa += 180
        if gal_re==-99.0:
            print 'NSA Re not available, 2MASS conversion not functional.'
            print 'Giving you a NaN, sorry.'
            gal_re = np.nan
    else:
        raise Exception('Invalid targets path.')
    return gal_center, gal_pa, gal_re

def get_friendly_ppxf_output(path):
    """
    Returns a friendly dict- and recarray-based set of ppxf output.
    See pPXFdriver.init_output_containers for details of fits file structure.
    """
    data, headers = utl.fits_quickread(path)
    friendly_data = {}

    # populate basic scalar parameters
    nbins = headers[0]['NAXIS2']
    nmoments = headers[0]['NAXIS1']
    npixels = headers[2]['NAXIS1']
    ntemps = headers[1]['NAXIS1']
    friendly_data['nmoments'] = nmoments
    friendly_data['nbins'] = nbins
    #friendly_data['npixels'] = npixels
    friendly_data['ntemps'] = ntemps
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

    # populate bin stuff
    dt = {'names':['id','chisq'],'formats':[int,np.float64]}
    friendly_data['bins'] = np.zeros((nbins,),dtype=dt)
    for ibin in range(nbins):
        friendly_data['bins'][ibin] = tuple(data[3][:2,ibin])

    # populate spectrum stuff
    dt = {'names':['bestmodel'], 'formats':['<f8']}
    friendly_data['spec'] = np.zeros((nbins,npixels),dtype=dt)
    friendly_data['spec']['bestmodel'] = data[2][0, ...]

    # populate waves 
    # this *should* match the bin spectra waves, but sometimes is off by 1
    friendly_data['waves'] = data[4]

    return friendly_data


def get_friendly_ppxf_output_mc(path):
    """
    Returns a friendly dict- and recarray-based set of ppxf mc output.
    See pPXFdriver.init_output_containers for details of fits file structure.
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

def friendly_temps(fits_path,temps_path):
    """
    Save template information in a nice friendly text file.
    All information is taken from the main fits file of the run, and only
     relevant information for template runs is saved to the new file.
    One template file for each bin is saved, since there can sometimes be
     multiple fullgalaxy bins. The bin number is stuck onto the generic
     temps_path just before the extension.
    """
    fitdata = get_friendly_ppxf_output(fits_path)
    ncols = len(fitdata['temps'].dtype.names)
    header = ('Columns are as follows:'
              '\n {colnames}'
              '\nMetadata is as follows:'
              '\n      nonzero templates: {ntemps_nonzero}'
              '\n out of total templates: {ntemps}'
              '\n       bin spectra file: PLACEHOLDER'
              '\n  bin spectra file date: PLACEHOLDER'
              '\n       output fits file: {fitsfile}'
              '\n  output fits file date: {fitsfiledate}'
              '\nThe best-fit moments are:'
              '\n {moments}'.format)
    colnames = ' '.join(fitdata['temps'].dtype.names)
    fitsfile = os.path.basename(fits_path)
    fitsfiledate = time.ctime(os.path.getmtime(fits_path))
    header=functools.partial(header,colnames=colnames,ntemps=fitdata['ntemps'],
                             fitsfile=fitsfile,fitsfiledate=fitsfiledate)
    for i in range(fitdata['nbins']):
        fmt = ['%i']
        fmt.extend(['%-8g']*(ncols-1))
        temps_root, temps_ext = os.path.splitext(temps_path)
        ii = np.nonzero(fitdata['temps']['weight'][i,:])[0]
        binpath = temps_root + str(fitdata['bins']['id'][i]) + temps_ext
        moments = ' '.join(['%-8g'%m for m in fitdata['gh']['moment'][i,:]])
        np.savetxt(binpath,fitdata['temps'][i,ii],fmt=fmt,
                   header=header(ntemps_nonzero=len(ii),moments=moments),
                   delimiter='\t')
    return
