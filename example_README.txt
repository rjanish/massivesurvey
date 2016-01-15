This is a standard readme, which should accompany all the galaxy kinematics I run. Ideally this will be updated with any changes to the file structure or pipeline, but let this serve as a disclaimer that this readme may not always match perfectly with the files it is supposed to describe. -Melanie

-----------------------------
Special notes for this galaxy:
-----------------------------
none

-------------------------------
General guide to the kinematics:
-------------------------------
The kinematics pipeline is divided into 4 scripts/steps. "s1", "s2", "s3", "s4" will show up in filenames to indicate which one the file is associated with. Here is a short description of each step and the output files it generates.
-s1: fits arc frames to get instrument resolution for each fiber vs wavelength
	-fibermaps.pdf: diagnostic plots
	-ir.txt: instrument resolution for each fiber at each arc line wavelength
-s2: bins the fibers spatially
	-bininfo.txt: bin centers, boundaries, etc
	-binmaps.pdf: diagnostic plots
	-binspectra.fits: spectra for each bin
	-fiberinfo.txt: mapping of fiber number to bin number
	-fullgalaxy.fits: spectra for the full galaxy (can be multiple spectra)
-s3: fits spectra using ppxf
	-main.fits: full fit results (note, also contains info in txt files below)
	-templates.pdf ("full", A runs only): diagnostic plots
	-temps-N.txt ("full", A runs only): templates for full galaxy spectrum N
	-moments.pdf ("bins", B runs only): diagnostic plots
	-moments.txt ("bins", B runs only): moments and errors for each bin
-s4: does miscellaneous postprocessing of the data
	-lambda.pdf: diagnostic plots
	-rprofiles.txt: radial profiles of various quantities

Each script has an associated parameter file, which contains the required input files and settings. Since some scripts are run multiple times, there are blocks of parameters in each file with different parameters for each run. (All but one block is commented out to run the script.)

The typical order of the pipeline is as follows:
-run s1 (mainrun)
-run s2 (fibers), iterating until bad_fiber_list.txt contains all fibers to remove
-run s3 (A-fibers) to get template lists
-run s3 (B-fibers), on only single-fiber bins, to find radial V for each fiber
-run s3X (B-fibers) to find radial V for each dither
-run s2 (folded) using results of s3X to align dithers before binning
-run s2X (on all s2 runs) if desired to inspect the shifted spectra up close
-run s3 (A-folded) to get better fullgalaxy kinematics, but do not redo template lists
-run s3 (B-folded) to get final kinematics on all bins
-run s4 (folded)

Some output files from the earlier stages (e.g. s2-fibers-binspectra.fits) may be deleted to save space.