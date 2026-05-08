import argparse
import os 

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.time import TimeDelta
from astropy.io import fits

from baseband import guppi
import baseband_tasks
from baseband_tasks.combining import Concatenate
from baseband_tasks import dm
from baseband_tasks.base import SetAttribute
from baseband_tasks.functions import Square
from baseband_tasks.integration import Integrate
from baseband_tasks.dispersion import DedisperseSamples, Dedisperse

from tools.tools import *
from tools.gp_finder import *

def Argument_Parser():
    parser = argparse.ArgumentParser(description='Input paramaters')

    parser.add_argument('--time_count', 
                        type=int,
                        help='Integer label to select one of 84 file-separated times; \
                        formatted to be used with jobarrays. Can only be a value from 0 to 167'
    )

    parser.add_argument('--gp_thresh',
                        default='5',
                        type=int,
                        help='Signal to noise cutoff. Defaults to 5.'
    )

    parser.add_argument('--DM',
                        default=56.7185,
                        type=float,
                        help='Dispersion measure'
    )

    parser.add_argument('--base_path',
                        default='/scratch/bscully/crab_GP_data_gbt/data/',
                        type=str,
                        help='Path to where the data is stored'
    )

    parser.add_argument('--results_path',
                        default='/scratch/bscully/crab_GP_data_gbt/results/',
                        type=str,
                        help="Path to where results should be sent"
    )

    parser.add_argument('--blur',
                        default=8,
                        type=int,
                        help='number of samples to integrate (blur) the data by')
    
    return parser.parse_args()


gp_width = (3 * u.us).to(u.s)

# unpack arguments
args = Argument_Parser()
time_count = args.time_count
gp_thresh = args.gp_thresh
DM = args.DM
base_path = args.base_path
results_path = args.results_path
blur = args.blur

# get the file labels
file_labels = time_count_to_file_labels(time_count)
is_first_half, start_time_label, scan_label, subint_label = file_labels

# prepare the data to be searched
squared, time_step, is_first_half, start_time = prep_data_from_file_labels(file_labels, base_path=base_path, DM=DM, blur=blur)

# search for the giant pulses
gp_times = np.array([])
gp_sns = np.array([])
is_end = False
squared.seek(0)
if not is_first_half:
    length = squared.shape[0]
    squared.seek(length // 2)
    # squared.seek(length - length // 128)  # reduced size for testing purposes
while not is_end:
    temp_gp_times, temp_gp_sns, is_end = search_time_chunk(squared, is_first_half, start_time, time_step, gp_thresh, samples=1000)
    gp_times = np.append(gp_times, temp_gp_times)
    gp_sns = np.append(gp_sns, temp_gp_sns)

# write data to fits file
if is_first_half:
    results_filename = os.path.join(results_path, "guppi_60631_{0}_DIAG_MESSIER1_{1}.{2}_0.fits".format(start_time_label, scan_label, subint_label))
else:
    results_filename = os.path.join(results_path, "guppi_60631_{0}_DIAG_MESSIER1_{1}.{2}_1.fits".format(start_time_label, scan_label, subint_label))

gp_times = [gp_time.value for gp_time in gp_times]

# Create columns for the FITS table
col1 = fits.Column(name='GP_TIMES', format='D', array=gp_times)
col2 = fits.Column(name='SNS', format='E', array=gp_sns)

table_hdu = fits.BinTableHDU.from_columns([col1, col2])

# Add some metadata to the header
table_hdu.header['COMMENT'] = 'Time and SNR data of giant pulses'

primary_hdu = fits.PrimaryHDU()
hdu_list = fits.HDUList([primary_hdu, table_hdu])
hdu_list.writeto(results_filename, overwrite=True)
