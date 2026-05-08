import os

import numpy as np
import astropy.units as u
from astropy.time import TimeDelta

from baseband import guppi
import baseband_tasks
from baseband_tasks.combining import Concatenate
from baseband_tasks import dm
from baseband_tasks.base import SetAttribute
from baseband_tasks.functions import Square
from baseband_tasks.integration import Integrate
from baseband_tasks.dispersion import DedisperseSamples, Dedisperse

def get_frequencies(files):
    """Get frequency list from set of baseband files"""
    
    base_freqs = [f.header0["OBSFREQ"] * u.MHz for f in files]
    chan_bw = files[0].header0["CHAN_BW"] * u.MHz
    n_chans = files[0].header0["OBSNCHAN"]
    obs_bw = files[0].header0["OBSBW"] * u.MHz
    band_size = chan_bw * n_chans
    total_chans = n_chans * len(files)

    freqs = np.concatenate([base_freq + np.linspace(0, n_chans-1, n_chans) * chan_bw for base_freq in base_freqs])
    freqs = freqs - obs_bw/2 + chan_bw/2 

    time_step = (1/chan_bw).to(u.s)
    return freqs, time_step

def time_count_to_file_labels(time_count):
    """ Get the numbers to find the correct filename depending on the count of time divisions you are at"""
    is_first_half = not (time_count % 2)
    if is_first_half:
        i = time_count // 2
        scan = (i // 14) + 8
        subint = i % 14
    else:
        i = (time_count - 1) // 2
        scan = (i // 14) + 8
        subint = i % 14
    start_time_label = (int(scan) - 8) * 309 + 7222

    start_time_label = f"{start_time_label:05}"
    scan_label = f"{scan:04}"
    subint_label = f"{subint:04}"

    return is_first_half, start_time_label, scan_label, subint_label


def search_time_chunk(data, is_first_half, start_time, time_step, gp_thresh, samples=1000):
    """
    Inputs:
        - data

        - is_first_half

        - start: start time of the larger time block (not this chunk) in mjd

        - time_step:

        - gp_thresh

        - samples

    Outputs:

    """
    gp_width = (3 * u.us).to(u.s)

    length = data.shape[0]
    if is_first_half:
        end = length // 2
        # end = length // 128  # for testing purposes
    else:
        end = length

    current = data.tell()  # start of this chunk (index not time)
    remaining = end - current
    
    # print("searching time chunk {0}, only {1} remaining".format(current, remaining))
    is_end = False
    # read the data and sum together the two polariazations
    if remaining <= samples:
        samples = remaining
        is_end = True

    d = (data.read(samples).T).sum(axis=1)

    d_avg_freqs = np.mean(d, axis=1)
    flat_d = d / d_avg_freqs[:, np.newaxis]  # dims are freq x time

    # sum along frequencies to get amplitude vs time
    summed_times = np.sum(flat_d, axis=0)
    sigma = np.std(summed_times)
    mean = np.mean(summed_times)

    # identify where the signal is above the threshold
    mask = np.argwhere(summed_times > (mean + gp_thresh*sigma))
    if len(mask) != 0:
        mask = np.reshape(mask, shape=(len(mask),))
    
    # remove redundant points (which are part of the same pulse)
    temp_mask = np.ones(mask.shape, dtype=bool)
    gp_width_samples = gp_width // time_step
    for i in range(len(mask)):
        if temp_mask[i]:
            temp_mask[(mask >= mask[i] - 5*gp_width_samples) & (mask <= mask[i] + 5*gp_width_samples) & (np.arange(len(mask)) != i)] = False
    mask = mask[temp_mask]

    # record timing of pulses
    if len(mask) != 0:
        print("trig! -- block {}".format(current))
        time_indices = mask + current
        times = time_indices * time_step  # in seconds
        time_deltas = TimeDelta(times, format='sec')

        time_deltas.format = 'jd'
        gp_total_times = start_time + time_deltas

        gp_sns = (summed_times[mask] - mean)/sigma  # signal to noise ratios    
        return gp_total_times, gp_sns, is_end
    else:
        return np.array([]), np.array([]), is_end


def prep_data_from_file_labels(file_labels, base_path, DM=56.7185, is_coherent_dedispersion=False, width=50*10**(-6) * u.s, blur=0):
    """Get data, get frequencies, concatenate, dedisperse, square, 
    and integrate if needed
    """
    is_first_half, start_time_label, scan_label, subint_label = file_labels

    # find corresponding files
    filenames0 = [os.path.join(base_path, "blc{0:02}_guppi_60631_{1}_DIAG_MESSIER1_{2}.{3}.raw".format(i, start_time_label, scan_label, subint_label)) for i in range(6)]
    filenames1 = [os.path.join(base_path, "blc{0:02}_guppi_60631_{1}_DIAG_MESSIER1_{2}.{3}.raw".format(i+10, start_time_label, scan_label, subint_label)) for i in range(6)]
    filenames2 = [os.path.join(base_path, "blc{0:02}_guppi_60631_{1}_DIAG_MESSIER1_{2}.{3}.raw".format(i+20, start_time_label, scan_label, subint_label)) for i in range(6)]
    filenames3 = [os.path.join(base_path, "blc{0:02}_guppi_60631_{1}_DIAG_MESSIER1_{2}.{3}.raw".format(i+30, start_time_label, scan_label, subint_label)) for i in range(8)]

    filenames = filenames0 + filenames1 + filenames2 + filenames3 

    # open files to read later
    fs = [guppi.open(filename) for filename in filenames]

    # define frequency range
    freqs, time_step = get_frequencies(fs)

    # reduce the samples per frame for easier reading later
    fs = [SetAttribute(f, samples_per_frame=1024) for f in fs]

    start_time = fs[0].start_time  # start time of the time block in mjd
    print("start time", start_time.iso)

    # Combine all files into one along the frequency axis
    data = Concatenate(fs, axis=2)
    data = SetAttribute(data, frequency=freqs, sideband=1)

    # use dispersion measure to dedisperse data
    dmeas = dm.DispersionMeasure(DM)
    dedispersed = DedisperseSamples(data, dmeas)

    # dedisperse data
    if is_coherent_dedispersion:
        dedispersed = Dedisperse(data, DM, frequency=freqs, sideband=1)
    else:
        dedispersed = DedisperseSamples(data, DM, frequency=freqs, sideband=1)

    squared = Square(dedispersed)

    if blur:
        squared = Integrate(squared, blur)
        time_step = time_step * blur

    return squared, time_step, is_first_half, start_time