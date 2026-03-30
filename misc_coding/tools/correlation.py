import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from baseband_tasks.functions import Square
from . import tools


def parabola(x, w, m, c):
    return w * (x - m)**2 + c

def line(x, m, c):
    return m * x + c

def find_peak_boundaries(time_stream, num_peaks):
    """
    Automatically detect boundaries between peaks in a time stream using peak detection.
    
    Uses scipy.signal.find_peaks to identify peaks, selects the num_peaks largest,
    then identifies the boundaries (valleys) between them.
    
    Parameters
    ----------
    time_stream : array-like
        The time-averaged signal showing peaks
    num_peaks : int
        The number of peaks expected in the data
    
    Returns
    -------
    boundaries : list
        Indices marking the boundaries between peaks (i0, i1, i2, ..., i_last)
        Length is num_peaks + 1
    """
    # Auto-detect prominence as fraction of signal range
    prominence = (np.max(time_stream) - np.min(time_stream)) * 0.1
    
    # Find peaks in the time_stream
    peaks, properties = find_peaks(time_stream, prominence=prominence)
    
    if len(peaks) < num_peaks:
        raise ValueError(f"Expected {num_peaks} peaks but found only {len(peaks)}")
    if len(peaks) > num_peaks:
        # If more peaks found, use the num_peaks largest ones
        peak_heights = time_stream[peaks]
        top_indices = np.argsort(peak_heights)[-num_peaks:]
        peaks = np.sort(peaks[top_indices])
    
    # Start with the beginning
    boundaries = []
    
    # Find valleys between consecutive peaks
    for i in range(len(peaks) - 1):
        # Find the minimum between consecutive peaks
        start_peak = peaks[i]
        end_peak = peaks[i + 1]
        valley_region = time_stream[start_peak:end_peak + 1]
        valley_idx = np.argmin(valley_region) + start_peak
        boundaries.append(valley_idx)
    # determine rough peak width
    if len(boundaries) > 1:
        mean_diff_boundaries = np.mean(np.diff(boundaries))
    else:
        mean_diff_boundaries = (boundaries[0] - peaks[0] ) + (peaks[1] - boundaries[0]) + 2
    # Add a boundary before the first peak and after the last peak
    boundaries.insert(0, max(0, peaks[0] - int(mean_diff_boundaries)))
    boundaries.append(min(len(time_stream), peaks[-1] + int(mean_diff_boundaries)))
    
    # Boundary to capture off-pulse region after last peak
    boundaries.append(boundaries[-1] + int(mean_diff_boundaries))

    return boundaries, peaks


def correlate_nanoshots(fs_data, blur, ts, freqs, num_peaks, plot=True):
    """
    Correlate nanoshot peaks automatically by detecting peak boundaries.
    
    Parameters
    ----------
    fs_data : array-like
        The frequency-time data from Square() processing, shape (freq, time)
    blur : int
        Blurring factor for frequency channel averaging
    ts : Quantity
        Time array with astropy units
    freqs : array-like
        Frequency array
    num_peaks : int
        Number of peaks to correlate
    plot : bool, optional
        Whether to generate plots (default True)
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'spectra': list of spectra for each peak and off-pulse region
        - 'correlations': dict of correlation arrays between peaks
        - 'peak_boundaries': list of indices marking peak boundaries
        - 'peak_power_idx': index of the brightest peak
        - 'time_stream': the time-averaged light curve
        - 'd0_blur': blurred data
        - 'freqs_blur': blurred frequency array
    """
    # Normalize data
    d0 = fs_data / np.sum(fs_data, axis=1, keepdims=True)
    time_stream = d0.sum(axis=0)
    
    # Apply blurring
    d0_blur = np.mean(np.reshape(d0, (-1, blur, d0.shape[1])), axis=1)
    freqs_blur = freqs[::blur]
    
    # Find peak boundaries automatically
    peak_boundaries, peaks = find_peak_boundaries(time_stream, num_peaks)
    peak_times = {}
    for i, peak in enumerate(peaks):
        peak_times[i] = ts[peak]

    # Extract data for each peak and off-pulse region
    peak_data = []
    for i in range(num_peaks):
        start_idx = peak_boundaries[i]
        end_idx = peak_boundaries[i + 1]
        peak_data.append((start_idx, end_idx))
    
    # Extract off-pulse region (after last peak)
    off_start = peak_boundaries[-2]
    off_end = peak_boundaries[-1]
    
    # Calculate spectra
    spectra = []
    for start_idx, end_idx in peak_data:
        peak_blur = d0_blur[:, start_idx:end_idx]
        spectrum = tools.get_spectrum(peak_blur)
        spectra.append(spectrum)
    
    off_blur = d0_blur[:, off_start:off_end]
    off_spectrum = tools.get_spectrum(off_blur)
    spectra.append(off_spectrum)
    
    # Find the brightest peak (by peak height, not integrated power)
    peak_power_idx = np.argmax(time_stream)
    peak_brightness = {i: np.max(time_stream[peak_boundaries[i]:peak_boundaries[i+1]]) 
                       for i in range(num_peaks)}
    brightest_peak = np.argmax([peak_brightness[i] for i in range(num_peaks)])
    
    # Diagnostic output
    # print(f"Brightest peak is peak {brightest_peak} (0 indexed) with power {peak_brightness[brightest_peak]:.2f}")
    
    spectra = [s - np.mean(s) for s in spectra]  # zero-mean spectra for correlation
    # Calculate correlations of brightest peak with others and off-pulse
    correlations = {}
    for i in range(num_peaks):
        key = f'corr_{brightest_peak}_{i}'
        correlations[key] = np.correlate(spectra[brightest_peak], spectra[i], mode='same')
    
    # Correlate brightest peak with off-pulse
    correlations[f'corr_{brightest_peak}_off'] = np.correlate(spectra[brightest_peak], spectra[-1], mode='same')

    # Create continuous correlation map with brightest peak
    peak_spectrum = d0_blur[:, peak_power_idx]
    peak_spectrum_sub = peak_spectrum - np.mean(peak_spectrum)  # zero-mean for correlation
    d0_blur_sub = d0_blur - np.mean(d0_blur, axis=0, keepdims=True)  # zero-mean for correlation
    cont_corr = np.array([np.correlate(peak_spectrum_sub, d0_blur_sub[:, i], mode='same') 
                          for i in range(d0_blur.shape[1])])
    # cont_corr = cont_corr / np.max(cont_corr, axis=1, keepdims=True)  # normalize each correlation by its max value
    
    # Plotting
    if plot:
        # Plot 1: Time stream with boundaries
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), height_ratios=[2, 1], sharex=True)
        axs[0].imshow(d0, aspect='auto', origin='lower')
        axs[0].set_ylabel('Frequency')
        axs[0].set_xlabel('Time index')
        axs[0].set_title('Dynamic Spectrum')
        axs[1].plot(time_stream)
        for boundary in peak_boundaries[:-1]:
            axs[1].axvline(boundary, color='r', linestyle='--', alpha=0.7)
        axs[1].set_ylabel('Power')
        axs[1].set_xlabel('Time index')
        axs[1].set_title('Time Stream with Peak Boundaries')

        
        plt.tight_layout()
        plt.show()
        
        # Plot 2: Individual spectra
        fig, axs = plt.subplots(num_peaks + 1, 1, figsize=(8, 6))
        if num_peaks == 1:
            axs = [axs]
        
        for i in range(num_peaks):
            axs[i].plot(freqs_blur, spectra[i])
            axs[i].set_title(f'Spectrum of peak {i}')
        axs[-1].plot(freqs_blur, spectra[-1])
        axs[-1].set_title('Spectrum of off-pulse region')
        plt.tight_layout()
        plt.show()
        
        # Plot 3: Correlations between peaks
        if num_peaks > 1:
            fig = plt.figure(figsize=(10, 6))
            for i, (key, corr_vals) in enumerate(correlations.items()):
                plt.plot(freqs_blur, corr_vals / np.max(np.abs(corr_vals)), label=key)
            # add vertical line at middle frequency value
            # plt.axvline(freqs_blur[len(freqs_blur) // 2], color='k', linestyle='--', alpha=0.7)
            plt.xlabel('Frequency (GHz)')
            plt.title("Correlation between peaks")
            plt.legend()
            plt.show()
        
        # Plot 4: Continuous correlation map
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 4), sharex=True, 
                                        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.15})
        
        im = ax0.imshow(cont_corr.T, aspect='auto', origin='lower',
                        extent=(ts[0].value, ts[-1].value, freqs_blur[0].value, freqs_blur[-1].value))
        ax1.plot(ts, time_stream)
        ax0.set_ylabel('Frequency (GHz)')
        ax1.set_xlabel('Time (ms)')
        
        fig.suptitle(f'Correlation between peak {brightest_peak} and entire pulse', y=0.995)
        fig.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.12)
        
        # Add colorbar below the plot to avoid affecting x-axis alignment
        cbar_ax = fig.add_axes([0.92, 0.35, 0.02, 0.4])
        plt.colorbar(im, cax=cbar_ax, label='Correlation')
        plt.show()
    
    return {
        'spectra': spectra,
        'correlations': correlations,
        'peak_times': peak_times,
        'peak_boundaries': peak_boundaries,
        'peak_power_idx': peak_power_idx,
        'brightest_peak': brightest_peak,
        'time_stream': time_stream,
        'd0_blur': d0_blur,
        'freqs_blur': freqs_blur,
        'continuous_correlation': cont_corr,
        'peak_data': peak_data,
        'off_pulse_region': (off_start, off_end)
    }

def find_means(correlations, ns_times, freqs_blur, n=11, plot=False, verbose=0):
    """
    inputs:
        correlations: dict of correlations where key is 'corr_[bright ns]_[other ns]'
        freqs_blur: dict blurred frequencies 
        ns_times: dict of ns times where key is the pulse number (consecutive in time)
        n: number of points to use for fitting parabola around each peak time
    """
    blur = 1664 // len(freqs_blur)
    mus = {}
    ws = {}
    r2_avg = 0
    for key in correlations.keys():
        # get the correlation widths from the autocorrelation
        if key[5] == key[7]:  # pick autocorrelation 
            auto_corr = correlations[key]
            middle = np.argmax(auto_corr)
            nearest_points = np.argsort(np.abs(np.arange(len(auto_corr)) - middle))[:n]
            nearest_points = np.sort(nearest_points)
            auto_corr[np.argmax(auto_corr)] = np.nan  # ignore centre value (perfect correlation)
            auto_corr /= np.nanmax(auto_corr)  # normalize

            # fit a parabola to the points around the middle of auto_corr
            x_data = freqs_blur[nearest_points].value
            y_data = auto_corr[nearest_points]
            popt_parabola, pcov_parabola = curve_fit(parabola, x_data, y_data, nan_policy='omit')

            unc_parabola = np.sqrt(np.diag(pcov_parabola))
            if verbose >= 2:
                print("Autocorrelation parabola parameters for {}: w = {:.3e} ± {:.3e}, m = {:.3e} ± {:.3e}".format(key,popt_parabola[0], unc_parabola[0], popt_parabola[1], unc_parabola[1]))

            w_nom = popt_parabola[0]
            w_nom_unc = unc_parabola[0]
            mus[key] = (popt_parabola[1], unc_parabola[1])
            ws[key] = (popt_parabola[0], unc_parabola[0])
            if plot:
                fig, axs = plt.subplots(2, 2, figsize=(8, 8))
                parabola_fit = parabola(freqs_blur[nearest_points].value, *popt_parabola)
                axs[0,0].plot(freqs_blur, auto_corr)
                axs[0,0].plot(freqs_blur[nearest_points], auto_corr[nearest_points], 'r')
                axs[0,0].plot(freqs_blur[nearest_points], parabola_fit, 'g--', label='Parabola Fit')
                axs[0,0].set_xlabel('Frequency (MHz)')
                axs[0,0].set_ylabel('Normalized Correlation')
                axs[0,0].set_title(f'Auto-correlation for {key}')
                plt.legend()
            # calculate the R2 of the parabola fit
            residuals = y_data - parabola(x_data, *popt_parabola)
            ss_res = np.nansum(residuals**2)
            ss_tot = np.nansum((y_data - np.nanmean(y_data))**2)
            r2 = 1 - (ss_res / ss_tot)
            r2_avg += r2

    plot_idx = 1
    for key in correlations.keys():
        if (key[5] == key[7]) or ("off" in key):
            continue
        else:
            cross_corr = correlations[key]
            # to find what should be the middle, find weighted average of frequencies using the cross_corr as weights, but only in the middle 50% of frequencies to avoid outliers dominating the mean
            middle_range = (freqs_blur.value > freqs_blur.value[int(len(freqs_blur)*0.33)]) & (freqs_blur.value < freqs_blur.value[int(len(freqs_blur)*0.66)])
            middle = np.argmin(np.abs(freqs_blur.value - np.average(freqs_blur.value[middle_range], weights=cross_corr[middle_range])))
            # or get middle based on highest point
            # middle = np.argmax(cross_corr)
            nearest_points = np.argsort(np.abs(np.arange(len(cross_corr)) - middle))[:n]
            nearest_points = np.sort(nearest_points)
            cross_corr /= np.nanmax(cross_corr)

            # fit a parabola to the points around the middle of cross_corr with w limited to w_nom ± N*w_nom_unc
            x_data = freqs_blur[nearest_points].value
            y_data = cross_corr[nearest_points]
            N = 2
            popt_parabola, pcov_parabola = curve_fit(parabola, x_data, y_data, p0=[w_nom, x_data[np.argmax(y_data)], np.nanmax(y_data)],
                                                    bounds=([w_nom - N*w_nom_unc, -np.inf, -np.inf], [w_nom + N*w_nom_unc, np.inf, np.inf]),
                                                        nan_policy='omit')
            unc_parabola = np.sqrt(np.diag(pcov_parabola))

            if verbose >= 2:
                print("Cross-correlation parabola parameters for {}: w = {:.3e} ± {:.3e}, m = {:.3e} ± {:.3e}".format(key, popt_parabola[0], unc_parabola[0], popt_parabola[1], unc_parabola[1]))

            mus[key] = (popt_parabola[1], unc_parabola[1])
            ws[key] = (popt_parabola[0], unc_parabola[0])
        
            if plot:
                parabola_fit = parabola(freqs_blur[nearest_points].value, *popt_parabola)
                axs[plot_idx//2, plot_idx%2].plot(freqs_blur, cross_corr)
                axs[plot_idx//2, plot_idx%2].plot(freqs_blur[nearest_points], cross_corr[nearest_points], 'r')
                axs[plot_idx//2, plot_idx%2].plot(freqs_blur[nearest_points], parabola_fit, 'g--', label='Parabola Fit')
                axs[plot_idx//2, plot_idx%2].set_xlabel('Frequency (MHz)')
                axs[plot_idx//2, plot_idx%2].set_ylabel('Normalized Correlation')
                axs[plot_idx//2, plot_idx%2].set_title(f'Cross-correlation for {key}')
                plot_idx += 1
            # calculate the R2 of the parabola fit
            residuals = y_data - parabola(x_data, *popt_parabola)
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_data - np.mean(y_data))**2)
            r2 = 1 - (ss_res / ss_tot)
            r2_avg += r2
    if plot:
        plt.tight_layout()
        plt.show()
    r2_avg /= len(correlations)
    if verbose >= 1:
        print(f"  - Average R2 of blur={blur} parabola fits: {r2_avg:.3f}")

    return mus, ws, ns_times, r2_avg


def get_correlation_peak_evolution(file, num_peaks, freqs, ts, blurs=[8, 16, 32, 64], plot=False, verbose=0):
    squared = Square(file)
    data0 = np.sum(squared.read(), axis=1).T

    r2s = np.zeros_like(blurs, dtype=float)
    mus_dict = {}
    ws_dict = {}
    ns_times_dict = {}

    broke = False
    if verbose >= 1:
        print("Trying different blurs to find best fit for peak frequency evolution...")
    for i, blur in enumerate(blurs):
        temp_results = correlate_nanoshots(data0, blur=blur, ts=ts, freqs=freqs, num_peaks=num_peaks, plot=plot)
        correlations = temp_results['correlations']
        freqs_blur = temp_results['freqs_blur']
        peak_times = temp_results['peak_times']

        n = int(np.ceil(11 * 16 / blur))
        mus, ws, ns_times, r2_avg = find_means(correlations, peak_times, freqs_blur, n=n, plot=plot, verbose=verbose)
        if r2_avg >= 0.35:
            print("Using results for blur = {}".format(blur))
            broke = True
            break
        r2s[i] = r2_avg
        mus_dict[i] = mus
        ws_dict[i] = ws
        ns_times_dict[i] = ns_times
    if broke:
        pass
    else:
        if verbose >= 1:
            print("None of the blurs gave a good fit (r2 >= 0.35). Returning results for blur with highest r2.")
        best_idx = np.argmax(r2s)
        print("Using results for blur = {}".format(blurs[best_idx]))
        mus = mus_dict[best_idx]
        ws = ws_dict[best_idx]
        ns_times = ns_times_dict[best_idx]
    
    ts = np.array([]) * u.ms
    ms_values = np.array([])
    ms_uncs = np.array([])
    for key in mus.keys():
        ns_num = int(key[7])
        ts = np.append(ts, ns_times[ns_num])
        ms_values = np.append(ms_values, mus[key][0])
        ms_uncs = np.append(ms_uncs, mus[key][1])

    sorted_indices = np.argsort(ts)
    ts = ts[sorted_indices].to(u.us)
    ms_values = ms_values[sorted_indices]
    ms_uncs = ms_uncs[sorted_indices]
    # fit to a line
    popt_line, pcov_line = curve_fit(line, ts.value, ms_values, sigma=ms_uncs, absolute_sigma=True)
    unc_line = np.sqrt(np.diag(pcov_line))
    if verbose >= 2:
        print("Line parameters: m = {} ± {}, b = {} ± {}".format(popt_line[0], unc_line[0], popt_line[1], unc_line[1]))
    line_fit = line(ts.value, *popt_line)

    eb = plt.errorbar(ts, ms_values, yerr=ms_uncs, fmt='o')
    colour = eb[0].get_color()

    # match plot colour to points)
    plt.plot(ts, line_fit, '--', c=colour, label='m = {:.3e} ± {:.3e} MHz/us'.format(popt_line[0], unc_line[0]))
    plt.xlabel(f'Time ({ts.unit})')
    plt.ylabel('Peak frequency of correlation (MHz)')
    plt.title('Peak frequency vs Time')
    plt.legend()
    # plt.show()  # can uncomment this if you want to do one fit per plot

    results = [ts, ms_values, ms_uncs]
    return results, popt_line, unc_line