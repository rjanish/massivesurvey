import numpy as np 
from astropy.io import fits 

raw_data_dir = "raw_data"
proc_data_dir = "proc_data"
science2list = "{}/2science_n1.list".format(raw_data_dir)
skylist = "{}/sky_n1.list".format(raw_data_dir)
skymodel_suffix = "pefy.fits" 
	# models of individual sky frames
subsci_suffix = "pefs.fits"   
	# science frames after sky subtraction and before cr masking

# get science, sky numbering and pairings 
skies = [] # coadded, sky0, sky1
with open(skylist, 'r') as skylistfile:
	for line in skylistfile:
		skyframes = line.split()[:3]
		skies.append(skyframes)
pairs = [] # science, coadded, sky0, sky1
with open(science2list, 'r') as science2listfile:
	for line in science2listfile:
		line = line.split()
		if len(line) == 3:
			sci_frame = line[0]
			coadded_sky = line[1]
			for setnum, [s0_plus_s1, s0, s1] in enumerate(skies):
				if coadded_sky == s0_plus_s1:
					pairs.append([sci_frame] + skies[setnum])

for sci, coadd, sky0, sky1 in pairs:
	# get sky models
	skyframes = ["{}/{}{}".format(proc_data_dir, s, skymodel_suffix)
					for s in [sky0, sky1]]
	skymodels = []
	for skyframe in skyframes:
		print 'reading skyframe {}'.format(skyframe)
		hdu = fits.open(skyframe)
		skymodels.append(hdu[0].data)
		hdu.close()
	skymodels = np.array(skymodels)
	# get science data
	hdu = fits.open("{}/{}{}".format(proc_data_dir, sci, 
									 subsci_suffix))
	scidata = hdu[0].data
	hdu.close()
	# reconstruct sci + sky
	unsub_science = scidata + skymodels[0,:,:] + skymodels[1,:,:]
	# make new subtracted frames
	for w0 in np.arange(0, 2.2, 0.2):
		w1 = 2.0 - w0
		new_coadd = w0*skymodels[0,:,:] + w1*skymodels[1,:,:]
		new_subtracted = unsub_science - new_coadd
		title = "skytest-{}-{}-{}.fits".format(sci, w0, w1)
		print 'writing trial subtraction {}, {}'.format(w0, w1)
		fits.writeto(title, new_subtracted, clobber=True)
