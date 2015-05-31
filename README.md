This package contains software used in the MASSIVE survey.

## Code Organization

The code is divided into a set executable scripts and a python package called
'massivepy'.

The software in massivepy is intended to be a set of building-blocks out of
which scripts that process and analyze IFU data can be easily constructed. It
contains modules that perform the heavy-lifting for common computations and
routines to simplify I/O operations. The modules are not executable and do not perform I/O directly nor work with data files on disk - that is left to
individual analysis scripts to allow more flexibility.

Calculations are done by executing the scripts found in bin/. These scripts
are indented to be small, a few hundred lines, and execute only one logical
piece of analysis. They will handle I/O of data and the logging of the analysis, while making calls to massivepy for significant calculation. The
purpose and usage of each of these scripts is described in its docstring and
command-line manual, viewable by running: $ python -h *path_to_script.py*.

## Requirements

### Directory Structure
As much as possible, the software tries to be agnostic about the lay-out of directories and data. This is achieved by a set of configuration files that
give paths to the locations of important dataset. But, **the location of the configuration files is hard-wired**. The directory structure is assumed to contain the following:

  massive/

    etc/

      datamap.txt

Analysis scripts are run from the top-level directory, here called *massive* though any name could be used.  The directory etc/ contains the configuration
files, and its name **is** assumed by the software. The module
massivepy.constants contains hard-coded paths to the conifg files in etc/,
specified relative to the top-level directory (massive/). To locate data,
the scripts in bin/ will use the massivepy.constants module to read the
needed entries from the config files in etc/.  The contents and format of
these config files is also assumed by the software, and is detailed below.

### Config Files
The config files are *dictionary files*, as read by the function *read_dict_file* in the python packing *utilities*.

The file etc/datamap.txt describes the locations of mostly-static data, and
should include the entries:
- template_libraries - points to the directory containing stellar template libraries
- raw_mitchell_cubes - points to the directory containing raw Mitchell IFU datacubes (i.e., the datacubes results from Jenny Greene's processing of Vaccine output)
proc_mitchell_cubes - points to the directory containing Mitchell IFU datacubes processed for MASSIVE analysis
target_positions - points to the file containing galaxy center coordinates and position angles
binned_mitchell -  points to the directory containing binned Mitchell IFU data

### Other Software
Required third-party public python packages:
- numpy
- scipy
- pandas
- matplotlib
- shapely
- descartes
- astropy

Required private python packages:
- utilities
- plotting
- fitting

Require public python scripts:
- ppxf
- mpfit

## Running Analysis

Analysis scripts are found in bin/, and should behave as proper unix
command-line programs. The calculations performed and usage of each script
is detailed in its help menu: $ python -h *path_to_script.py*.

The scripts in bin/ are indented to be run from the top-level directory (see above), regardless of the actual location of bin/.