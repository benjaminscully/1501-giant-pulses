import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from scipy.signal import find_peaks

def get_spectrum(data, normalize=False):
    """data should be time vs frequency spectrograph"""
    if normalize:
        avg_along_f = np.mean(data, axis=1)  # average along frequency channels
        data = data / avg_along_f[:, np.newaxis]
    
    sum_along_time = np.sum(data, axis=0)  # sum along time channels -- gives time stream
    sum_along_time = np.abs(sum_along_time - np.mean(sum_along_time))
    time_weights = sum_along_time/np.max(sum_along_time)  # normalize time stream to treat as weights

    spectrum = np.dot(data, time_weights)
    # normalize spectrum by number of time samples to get average power per frequency channel
    spectrum /= data.shape[0]

    return spectrum

def make_triple_plot(spect, ts, freqs, width=50*u.us, interp="auto", cmap="magma", figsize=(5,8), vlim=0):
    flat_d = spect
    summed_times = np.sum(flat_d, axis=0)  # summed over times
    spectrum = get_spectrum(flat_d)

    freqs = freqs.to(u.GHz)
    f_min = np.min(freqs)
    f_max = np.max(freqs)

    f, axs = plt.subplots(2,2, width_ratios = [6, 2], height_ratios= [1, 4],
                        sharex="col", sharey=False, figsize=figsize)
    axs[1,1].sharey(axs[1,0])
    
    axs[0,0].plot(ts, summed_times)
    axs[1,1].plot(spectrum, freqs)

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