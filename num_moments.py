"""
This script is designed to test the num_moments input to ppxf
"""

import argparse
import io as mpio

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('paramfiles', nargs='*', type=str, 
                    help='path(s) to parameter files')
args = parser.parse_args()
all_paramfile_paths = args.paramfiles


