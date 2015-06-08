This package contains software used in the MASSIVE survey.


## Code Organization

The code is divided into a set executable scripts and a python package called
'massivepy'.

The software in massivepy is intended to be a set of building-blocks out of
which scripts that process and analyze IFU data can be easily constructed. It
contains modules that perform the heavy-lifting for common computations and
routines to simplify I/O operations. The modules are not executable and do not perform I/O directly nor work with data files on disk - that is left to
individual analysis scripts to allow more flexibility.

Calculations are done by executing the scripts found in the main directory.
(Additional scripts for comparison and testing purposes are found in bin/.)
These scripts
are indented to be small, a few hundred lines, and execute only one logical
piece of analysis. They will handle I/O of data and the logging of the analysis, while making calls to massivepy for significant calculation. The
purpose and usage of each of these scripts is described in its docstring and
command-line manual, viewable by running: $ python *path_to_script.py* -h.

## Requirements

### Directory Structure
As much as possible, the software tries to be agnostic about the lay-out of directories and data. This is achieved by a set of parameter files for each script that
give paths to the locations of important dataset.
Parameter files can contain relative (to the location the scripts are run from) or absolute paths.
(Absolute paths should allow you to run the script from any location.)

(Previous versions required the following "global" parameter file, but this should no longer be necessary.
The location of this parameter file was hard-coded into massivepy.constants.):
- massive/etc/datamap.txt

### Input Files
The input parameter files are dictionary files, as read by the function read_dict_file in the python package utilities.
Each script in the main directory has an example input parameter file; the easiest way to run the scripts is to copy the example files to some directory of your choice, then rename and edit them as necessary.
Include the galaxy name in the parameter file names; each script can accept multiple parameter files as command line input in order to process multiple galaxies at once.

### Other Software
Required third-party public python packages (these packages need to reside
in directories on the PYTHONPATH):
- numpy
- scipy
- pandas
- matplotlib
- shapely
- descartes
- astropy

Required private python packages (these packages need to reside in
directories on the PYTHONPATH):
- utilities
- plotting
- fitting

Require public python scripts (these are not packages, but collections of
scripts - the directories that hold each such collection must itself be
on the PYTHONPATH):
- ppxf
- mpfit

## Usage

Analysis scripts should behave as proper unix command-line programs.
The calculations performed and usage of each script
is detailed in its help menu: $ python *path_to_script.py* -h.
Each script takes one or more parameter files as command line arguments, with all needed inputs specified in the parameter files.