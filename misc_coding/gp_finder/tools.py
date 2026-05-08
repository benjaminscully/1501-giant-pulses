import numpy as np
import os

from astropy.time import Time, TimeDelta
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation

from baseband import guppi
import baseband_tasks
from baseband_tasks.combining import Concatenate
from baseband_tasks import dm
from baseband_tasks.base import SetAttribute
from baseband_tasks.functions import Square
from baseband_tasks.integration import Integrate
from baseband_tasks.dispersion import DedisperseSamples, Dedisperse
from baseband_tasks.channelize import Dechannelize
from baseband_tasks.base import Task

import matplotlib.pyplot as plt

def time_to_file_labels(time):
    """Find the time labels of the file containing `time'
    Inputs:
        - time: the time you are looking for in 'mjd' format

    Outputs:
        - start_time_label
        - scan_label
        - subint_label
        - time_offset: the time offset from the start of the corresponding file
    """
    if not isinstance(time, Time):
        time = Time(time, format='mjd')

    start_time = 60631.083587962959427  # start time in mjd -- corresponds to 2024-11-17 02:00:22.00000 iso

    time_block_long = 0.000265121438025 # mjd -> 22.9064922453  # seconds
    time_block_short = 0.000129810194567 # mjd -> 11.2156008106  # seconds

    rnge = np.linspace(0, 83, 84)  # number of different time files
    time_edges = np.zeros_like(rnge)
    for i in range(len(time_edges)):
        if i==0:
            time_edges[i] = 0
        elif (i%14==0) & (i!=0):
            time_edges[i] = time_edges[i-1] + time_block_short
        else:
            time_edges[i] = time_edges[i-1] + time_block_long
    
    time_edges = time_edges + start_time
    time_edges = Time(time_edges, format='mjd', precision=9)

    mask = np.where(time > time_edges)
    found_idx = mask[0][-1]
    found_time_edge = time_edges[found_idx]
    time_offset = TimeDelta(time - found_time_edge, format='jd')

    scan = (found_idx // 14) + 8
    subint = found_idx % 14
    start_time_label = (int(scan) - 8) * 309 + 7222

    start_time_label = f"{start_time_label:05}"
    scan_label = f"{scan:04}"
    subint_label = f"{subint:04}"

    return start_time_label, scan_label, subint_label, time_offset


def get_frequencies(files):
    """Get frequency list from set of baseband files"""
    
    base_freqs = [f.header0["OBSFREQ"] * u.MHz for f in files]
    chan_bw = files[0].header0["CHAN_BW"] * u.MHz
    n_chans = files[0].header0["OBSNCHAN"]
    obs_bw = files[0].header0["OBSBW"] * u.MHz
    band_size = chan_bw * n_chans
    total_chans = n_chans * len(files)

    # freqs = np.array([])
    # for base_freq in base_freqs:
    #     freqs_1 = base_freq + np.linspace(0, (n_chans//2) - 1, n_chans//2) * chan_bw
    #     freqs_2 = np.flip(base_freq - np.linspace(0, (n_chans//2) - 1, n_chans//2) * chan_bw)
    #     temp_freqs = np.concatenate((freqs_1, freqs_2))
    #     freqs = np.concatenate((freqs, temp_freqs))

    freqs = np.concatenate([base_freq + np.linspace(0, n_chans-1, n_chans) * chan_bw for base_freq in base_freqs])
    freqs = freqs - obs_bw/2 + chan_bw/2 
    
    # freqs = np.array([base_freq + np.linspace(0, n_chans-1, n_chans) * chan_bw for base_freq in base_freqs])
    # freqs = np.reshape(freqs, freqs.shape[0]*freqs.shape[1])
    # freqs = freqs
    return freqs


def data_from_time(time, DM=56.7185, is_coherent_dedispersion=False,
                    width=50*10**(-6) * u.s, blur=0, dechannelize=False, n=None):
    """Get data from a time in 'mjd' format"""
    if not isinstance(time, Time):
        time = Time(time, format='mjd', precision=9)
    
    file_labels = time_to_file_labels(time)

    # flat_d, d, ts, freqs = data_from_file_labels(file_labels, DM, is_coherent_dedispersion, width, blur, dechannelize, n)

    if dechannelize:
        dechannelized, flat_d, d, ts, freqs = data_from_file_labels(file_labels, DM, is_coherent_dedispersion, width, blur, dechannelize, n)
        return dechannelized, flat_d, d, ts, freqs
    else:
        flat_d, d, ts, freqs = data_from_file_labels(file_labels, DM, is_coherent_dedispersion, width, blur, dechannelize, n)
        return flat_d, d, ts, freqs


def data_from_file_labels(file_labels, DM=56.7185, is_coherent_dedispersion=False,
                             width=50*10**(-6) * u.s, blur=0, dechannelize=False, n=None):
    
    start_time_label, scan_label, subint_label, time_offset = file_labels

    data_path = "/scratch/bscully/crab_GP_data_gbt/data/"

    filenames0 = [os.path.join(data_path, "blc{0:02}_guppi_60631_{1}_DIAG_MESSIER1_{2}.{3}.raw".format(i, start_time_label, scan_label, subint_label)) for i in range(6)]
    filenames1 = [os.path.join(data_path, "blc{0:02}_guppi_60631_{1}_DIAG_MESSIER1_{2}.{3}.raw".format(i+10, start_time_label, scan_label, subint_label)) for i in range(6)]
    filenames2 = [os.path.join(data_path, "blc{0:02}_guppi_60631_{1}_DIAG_MESSIER1_{2}.{3}.raw".format(i+20, start_time_label, scan_label, subint_label)) for i in range(6)]
    filenames3 = [os.path.join(data_path, "blc{0:02}_guppi_60631_{1}_DIAG_MESSIER1_{2}.{3}.raw".format(i+30, start_time_label, scan_label, subint_label)) for i in range(8)]

    filenames = filenames0 + filenames1 + filenames2 + filenames3

    fs = [guppi.open(filename) for filename in filenames]

    freqs = get_frequencies(fs)
    # argsort_freqs = np.argsort(freqs)
    # freqs = freqs[argsort_freqs]
    chan_bw = fs[0].header0["CHAN_BW"] * u.MHz
    time_step = (1/chan_bw).to(u.s)

    fs = [SetAttribute(f, samples_per_frame=1024) for f in fs]
    data = Concatenate(fs, axis=2)
    # data = Task(data, lambda x: x[:, :, argsort_freqs])
    data = SetAttribute(data, frequency=freqs, sideband=1)

    if is_coherent_dedispersion:
        dedispersed = Dedisperse(data, DM, frequency=freqs, sideband=1)
    else:
        dedispersed = DedisperseSamples(data, DM, frequency=freqs, sideband=1)

    if dechannelize:
        dechannelized = Dechannelize(dedispersed, n=n)

    squared = Square(dedispersed)

    if blur:
        squared = Integrate(squared, sample_blur)
    
    time = time_offset.sec * u.s

    start_frac = 0.2
    samples = int(width // time_step)
    # squared.seek(time - (width)/2)  
    squared.seek(time - ((start_frac * width)))  
    d = (squared.read(samples).T)
    d = d.sum(axis=1) # sum polarizations together
    d_avg_freqs = np.mean(d, axis=1)
    flat_d = d / d_avg_freqs[:, np.newaxis]
    summed_times = np.sum(flat_d, axis=0)

    # t0 = 0 * u.s
    # tf = width
    t0 = -start_frac * width
    tf = (1 - start_frac) * width
    ts = np.linspace(t0.to(u.us), tf.to(u.us), len(summed_times))

    if dechannelize:
        return dechannelized, flat_d, d, ts, freqs
    else:
        return flat_d, d, ts, freqs


def get_spectrum(data, normalize=False, subtract=False, mean=False):
    """data should be time vs frequency spectrograph"""
    if normalize:
        avg_along_f = np.mean(data, axis=1)  # average along frequency channels
        data = data / avg_along_f[:, np.newaxis]
    if subtract:
        mean = np.mean(data)
        data = data - mean
    
    sum_along_time = np.sum(data, axis=0)  # sum along time channels -- gives time stream
    sum_along_time = np.abs(sum_along_time - np.mean(sum_along_time))
    time_weights = sum_along_time/np.max(sum_along_time)  # normalize time stream to treat as weights

    spectrum = np.dot(data, time_weights)
    if mean:
        spectrum = spectrum / len(time_weights)

    return spectrum


def make_phase_profile(times, telescope_loc=EarthLocation.of_site("Green Bank Telescope"),
                        object_loc=SkyCoord.from_name("Crab Pulsar"), freq=0):
    
    times = Time(times, format="mjd", precision=9, location=telescope_loc)

    if not freq:
        # !linearly interpolate between to edge times to the start time!
        freq_nov = 29.5552429346 * u.Hz  # Nov 15 2024 -- 60629 0.009517
        freq_dec = 29.5542931002 * u.Hz  # Dec 15 2024 -- 60659  0.020752

        t_diff_nov_dec = 60659.020752 - 60629.009517
        t_diff_nov_start = 60631.07222 - 60629.009517
        f_diff_nov_dec = freq_nov - freq_dec
        f_offset = t_diff_nov_start/t_diff_nov_dec * f_diff_nov_dec
        freq = freq_nov - f_offset  # crab pulsar's frequency on Nov 17 at 07222 s

    t = times + times.light_travel_time(object_loc)
    
    a = ((t.tdb - t.tdb[0]) * freq)%1

def plot_gp_time(time, save_file, DM=56.7185, is_coherent_dedispersion=False,
                width=50*10**(-6) * u.s, sn=0, interp='auto', figsize=(5,5)):
    flat_d, d, ts, freqs = data_from_time(time, DM=DM, is_coherent_dedispersion=is_coherent_dedispersion, width=width)
    summed_times = np.sum(flat_d, axis=0)  # summed over times
    summed_freqs = np.sum(flat_d, axis=1)  # summed over frequencies
    
    f, axs = plt.subplots(2, 1, sharex=True, height_ratios=(4, 1), figsize=figsize)   
    
    axs[1].plot(ts, summed_times)
    im = axs[0].imshow(flat_d, aspect="auto", extent=(ts[0].value, ts[-1].value,
                            freqs[0].value, freqs[-1].value), interpolation=interp, origin="lower")
    # axs[1,0].hlines(freqs[::64], t0.value, tf.value, color="orange")

    axs[1].set_xlabel("us")
    axs[0].set_ylabel("GHz")
    
    if sn:
        f.suptitle(f"GP with S/N {sn:.3f} at {time.value:.8f}")
    else:
        f.suptitle(f"GP at {time.value:.8f}")
    f.tight_layout()
    plt.savefig(save_file)
    plt.show()
    return 

def plot_gp_data(data, ts, freqs, save_file, sn=0, interp='auto', figsize=(5,5)):
    summed_times = np.sum(data, axis=0)
    f, axs = plt.subplots(2, 1, sharex=True, height_ratios=(4, 1), figsize=figsize) 
    
    axs[1].plot(ts, summed_times)
    im = axs[0].imshow(data, aspect="auto", extent=(ts[0].value, ts[-1].value,
                            freqs[0].value, freqs[-1].value), interpolation=interp, origin="lower")
    # axs[1,0].hlines(freqs[::64], t0.value, tf.value, color="orange")

    axs[1].set_xlabel("us")
    axs[0].set_ylabel("GHz")
    
    if sn:
        f.suptitle(f"GP with S/N {sn:.3f}")
    else:
        f.suptitle(f"GP with unknown S/N")
    f.tight_layout()
    plt.savefig(save_file)
    plt.show()
    return

def make_triple_plot(spect, ts, freqs, width=50*u.us, interp="auto", cmap="magma", figsize=(5,8), vlim=0, savefig=False, savepath="triple_plot.png"):
    flat_d = spect
    summed_times = np.sum(flat_d, axis=0)  # summed over times
    spectrum = get_spectrum(flat_d)

    freqs = freqs.to(u.GHz)
    f_min = np.min(freqs)
    f_max = np.max(freqs)

    ts = ts.to(u.us)

    f, axs = plt.subplots(2,2, width_ratios = [6, 2], height_ratios= [1, 4],
                        sharex="col", sharey=False, figsize=figsize, gridspec_kw={'wspace': 0, 'hspace': 0})
    axs[1,1].sharey(axs[1,0])
    
    axs[0,0].plot(ts, summed_times, c='k')
    axs[1,1].plot(spectrum, freqs, c='k')

    if vlim:
        mx = np.max(flat_d)
        vmax = vlim * np.max(flat_d)
        im = axs[1, 0].imshow(flat_d, aspect="auto", extent=(ts[0].value, ts[-1].value,
                                freqs[0].value, freqs[-1].value), interpolation=interp, origin="lower",
                                cmap=cmap, vmax=vmax)
    else:
        im = axs[1, 0].imshow(flat_d, aspect="auto", extent=(ts[0].value, ts[-1].value,
                                f_min.value, f_max.value), interpolation=interp, origin="lower",
                                cmap=cmap)

    axs[0,1].axis("off")
    axs[1,1].tick_params(labelleft=False)

    axs[1,0].set_xlabel("us")
    axs[1,0].set_ylabel("GHz")

    f.tight_layout()
    plt.show()
    if savefig:
        f.savefig(savepath, bbox_inches='tight')


