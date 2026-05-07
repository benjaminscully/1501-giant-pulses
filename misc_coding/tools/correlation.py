from statistics import correlation

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy import units as u
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from baseband_tasks.functions import Square
from . import tools

# ── Curve shapes used to fit correlation profiles ─────────────────────────

def parabola(x, w, m, A):
    return -w**2 * (x - m)**2 + A

def line(x, m, c):
    return m * x + c

def gaussian(x, sigma, mu, A):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def triangular(x, w, m, A):
    res = np.zeros_like(x)
    res[x < m]  =  w * (x[x < m]  - m) + A
    res[x >= m] = -w * (x[x >= m] - m) + A
    return res


def _compute_r2(y_data, y_fit, use_nan=False):
    """Coefficient of determination (R²) for a model fit."""
    if use_nan:
        ss_res = np.nansum((y_data - y_fit)**2)
        ss_tot = np.nansum((y_data - np.nanmean(y_data))**2)
    else:
        ss_res = np.sum((y_data - y_fit)**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
    return 1.0 - ss_res / ss_tot


def find_peak_boundaries(time_stream, num_peaks):
    """
    Detect boundaries between peaks in a time stream.

    Uses scipy.signal.find_peaks to identify peaks, selects the
    ``num_peaks`` largest, then locates the valley between each pair of
    consecutive peaks.  Two extra boundaries are appended so the final
    region captures an off-pulse window after the last peak.

    Parameters
    ----------
    time_stream : ndarray, shape (n_time,)
        Time-averaged signal.
    num_peaks : int
        Expected number of peaks.

    Returns
    -------
    boundaries : list of int
        Length ``num_peaks + 2``.  Slices ``[boundaries[i]:boundaries[i+1]]``
        isolate each peak; ``[boundaries[-2]:boundaries[-1]]`` is off-pulse.
    peaks : ndarray of int
        Indices of the ``num_peaks`` selected peak maxima.
    """
    prominence = (np.max(time_stream) - np.min(time_stream)) * 0.1
    peaks, _   = find_peaks(time_stream, prominence=prominence)

    if len(peaks) < num_peaks:
        raise ValueError(f"Expected {num_peaks} peaks but found only {len(peaks)}")
    if len(peaks) > num_peaks:
        top_idx = np.argsort(time_stream[peaks])[-num_peaks:]
        peaks   = np.sort(peaks[top_idx])

    #  ===== = ORIGINAL: define boundary as minimum between peaks, with padding to capture off-pulse window ======
    # # Valley between each consecutive pair of peaks defines a boundary
    # boundaries = []
    # for i in range(len(peaks) - 1):
    #     valley_region = time_stream[peaks[i]:peaks[i + 1]]
    #     boundaries.append(int(np.argmin(valley_region)) + peaks[i])

    # # Characteristic inter-peak spacing, used to pad before/after the pulse
    # if len(boundaries) > 1:
    #     pad = int(np.mean(np.diff(boundaries)))
    # else:
    #     pad = int((boundaries[0] - peaks[0]) + (peaks[1] - boundaries[0])) + 2

    # boundaries.insert(0, max(0, peaks[0] - pad))
    # boundaries.append(min(len(time_stream) - pad, peaks[-1] + pad))
    # boundaries.append(boundaries[-1] + pad)   # off-pulse window end

    # boundaries = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    #  ====== ALTERNATIVE: take the two adjacent points to each peak as boundaries, then add an off-pulse window after the last peak ======
    boundaries = []
    for peak in peaks:
        start = peak - 1
        end = peak + 1
        boundaries.append((max(0, start), min(len(time_stream) - 1, end)))
    boundaries.append((boundaries[-1][1] + 10, boundaries[-1][1] + 15))  # off-pulse window after last peak

    return boundaries, peaks


# ── Helpers called by correlate_nanoshots ─────────────────────────────────

def _blur(fs_data, blur, freqs):
    """
    Compute``fs_data`` time-stream, and average
    adjacent frequency channels by a factor of ``blur``.
    Effecitvely blurs in frequency.

    Parameters
    ----------
    fs_data : ndarray, shape (n_freq, n_time)
    blur    : int
    freqs   : Quantity, shape (n_freq,)

    Returns
    -------
    d0          : ndarray, shape (n_freq, n_time)   — row-normalised
    d0_blur     : ndarray, shape (n_freq//blur, n_time)
    time_stream : ndarray, shape (n_time,)
    freqs_blur  : Quantity, shape (n_freq//blur,)
    """
    
    d0 = fs_data
    time_stream = d0.sum(axis=0)
    d0_blur     = np.mean(np.reshape(d0, (-1, blur, d0.shape[1])), axis=1)
    freqs_blur  = freqs[::blur]
    return d0, d0_blur, time_stream, freqs_blur


def _extract_and_normalize_spectra(d0_blur, peak_boundaries, off_start, off_end):
    """
    Extract a time-averaged spectrum for each peak window and the
    off-pulse region, then normalise relative to the off-pulse.

    Normalisation convention
    ------------------------
    Peak spectra  : (spectrum - off_spectrum) / off_spectrum
    Off-pulse     : (off_spectrum - mean(off_spectrum)) / off_spectrum

    Parameters
    ----------
    d0_blur   : ndarray, shape (n_freq_blur, n_time)
    peak_boundaries : list of (start_idx, end_idx) tuples
    off_start : int
    off_end   : int

    Returns
    -------
    spectra : list of ndarray, length len(peak_boundaries) + 1
        Last element is the normalised off-pulse spectrum.
    """
    spectra      = [tools.get_spectrum(d0_blur[:, s:e + 1]) for s, e in peak_boundaries] # extract spectra for each peak window, end inclusive
    # off_spectrum = np.mean(d0_blur[:, off_start:off_end], axis=1)
    # spectra.append(off_spectrum)
    
    spectra = [s - np.mean(s) for s in spectra]  # zero-mean for correlation
    
    return spectra


def _find_brightest_peak(time_stream, peak_boundaries, num_peaks):
    """Return the 0-based index of the peak with the highest maximum."""
    
    peak_max = [
        np.max(time_stream[peak_boundaries[i][0]:peak_boundaries[i][1] + 1])  # max within each peak boundary, end inclusive
        for i in range(num_peaks)
    ]
    return int(np.argmax(peak_max))


def _compute_peak_correlations(spectra, brightest_peak, num_peaks):
    """
    Cross-correlate the brightest peak's spectrum with every other peak
    spectrum and with the off-pulse spectrum.

    All correlations are divided by the auto-correlation of a flat
    (all-ones) array to provide a consistent length-independent scale.

    Returns
    -------
    correlations   : dict  {key: ndarray}
        Keys: ``'corr_{brightest}_{i}'`` and ``'corr_{brightest}_off'``.
    norm_auto_corr : ndarray — the normalisation denominator.
    """
    norm_array      = np.ones_like(spectra[0])
    norm_auto_corr  = np.correlate(norm_array, norm_array, mode='same')
    bright_spectrum = spectra[brightest_peak]

    correlations = {}
    for i in range(num_peaks):
        key = f'corr_{brightest_peak}_{i}'
        correlations[key] = np.correlate(bright_spectrum, spectra[i], mode='same') / norm_auto_corr

    # Correlation with off-pulse spectrum
    correlations[f'corr_{brightest_peak}_off'] = (
        np.correlate(bright_spectrum, spectra[-1], mode='same') / norm_auto_corr
    )
    return correlations, norm_auto_corr


def _compute_continuous_correlation(d0_blur, peak_power_idx, norm_auto_corr):
    """
    Correlate the spectrum at the sample of peak power against every
    time sample in ``d0_blur``, producing a 2-D correlation map.

    Parameters
    ----------
    d0_blur        : ndarray, shape (n_freq_blur, n_time)
    peak_power_idx : int — time index of the peak-power sample
    norm_auto_corr : ndarray — normalisation from ``_compute_peak_correlations``

    Returns
    -------
    cont_corr : ndarray, shape (n_time, n_freq_blur)
    """
    peak_spectrum_sub = d0_blur[:, peak_power_idx] - np.mean(d0_blur[:, peak_power_idx])
    d0_blur_sub       = d0_blur - np.mean(d0_blur, axis=0, keepdims=True)
    return np.array([
        np.correlate(peak_spectrum_sub, d0_blur_sub[:, i], mode='same') / norm_auto_corr
        for i in range(d0_blur.shape[1])
    ])


def correlate_nanoshots(fs_data, blur, ts, freqs, num_peaks, plot=True, avg_spectrum=None):
    """
    Build spectra and spectral correlations for all nanoshot peaks
    in a single frequency-time observation.

    Parameters
    ----------
    fs_data      : ndarray, shape (n_freq, n_time)
        Frequency-time data from Square() processing.
    blur         : int
        Number of frequency channels to average together.
    ts           : Quantity, shape (n_time,)
        Time axis.
    freqs        : Quantity, shape (n_freq,)
        Frequency axis.
    num_peaks    : int
        Number of nanoshot peaks expected.
    plot         : bool, optional
        Whether to produce diagnostic plots.  Default: True.
    avg_spectrum : ndarray, shape (2, n_freq,)
        Optional average spectrum across all files, used to further normalise each peak spectrum.


    Returns
    -------
    dict with keys:
        'spectra'               — list of normalised spectra (num_peaks + 1)
        'correlations'          — {key: ndarray} spectral correlations
        'peak_times'            — {peak_idx: Quantity time}
        'peak_boundaries'       — list of (start, end) tuples of pulse windows and noise window
        'peak_power_idx'        — int, sample index of peak power
        'brightest_peak'        — int, index of the brightest peak
        'time_stream'           — ndarray, time-averaged light curve
        'd0_blur'               — ndarray, blurred normalised data
        'freqs_blur'            — Quantity, blurred frequency axis
        'continuous_correlation'— ndarray, shape (n_time, n_freq_blur)
        'off_pulse_region'      — (off_start, off_end)
    """
    # 1. Frequency-blur the data
    _, d0_blur, time_stream, freqs_blur = _blur(fs_data, blur, freqs)

    # 2. Locate peaks and define time windows
    peak_boundaries, peaks = find_peak_boundaries(time_stream, num_peaks)
    peak_times  = {i: ts[pk] for i, pk in enumerate(peaks)}
    off_start   = peak_boundaries[-1][0]  # start of off-pulse region is end of last peak boundary
    off_end     = peak_boundaries[-1][1]  # end of off-pulse region is end of last peak boundary + pad

    # 3. Extract and normalise spectra
    # spectra = _extract_and_normalize_spectra(d0_blur, peak_boundaries, off_start, off_end)
    spectra      = [tools.get_spectrum(d0_blur[:, s:e + 1]) for s, e in peak_boundaries] # extract spectra for each peak window, end inclusive

    
    if avg_spectrum is not None:
        temp_avg_spectrum, nom_freqs = avg_spectrum
        temp_avg_spectrum = np.interp(freqs_blur.value, nom_freqs.value, temp_avg_spectrum)  # interpolate average spectrum to blurred frequencies
        spectra = [s / temp_avg_spectrum for s in spectra]  # further normalise by average spectrum across files
    
    spectra = [s - np.mean(s) for s in spectra]  # zero-mean for correlation

    # 4. Identify brightest peak; compute spectral correlations
    brightest_peak = _find_brightest_peak(time_stream, peak_boundaries, num_peaks)
    peak_power_idx = int(np.argmax(time_stream))
    correlations, norm_auto_corr = _compute_peak_correlations(spectra, brightest_peak, num_peaks)

    # 5. Build continuous correlation map
    cont_corr = _compute_continuous_correlation(d0_blur, peak_power_idx, norm_auto_corr)

    # if plot:  # plot the time stream with peak boundaries
    #     print("plotting time stream with peak boundaries...")
    #     plt.figure(figsize=(10, 4))
    #     plt.plot(ts, time_stream, label='Time Stream')
    #     # for i in range(len(peak_boundaries)):  # draw lines at the boundaries of each peak region
    #     #     plt.axvline(ts[peak_boundaries[i]].value, color='red', linestyle='--', alpha=0.7)
    #     for i in range(len(peak_boundaries) - 1):  # draw lines at the boundaries of each peak region
    #         start, end = peak_boundaries[i]
    #         plt.axvline(ts[start].value, color='red', linestyle='--', alpha=0.7)
    #         plt.axvline(ts[end].value, color='red', linestyle='--', alpha=0.7)

    #     plt.axvspan(ts[off_start].value, ts[off_end].value, alpha=0.3, color='gray', label='Off-pulse')
    #     plt.xlabel(f'Time ({ts.unit})')
    #     plt.ylabel('Normalised Power')
    #     plt.title('Time Stream with Peak Boundaries')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()

    if plot:  # plot the individual spectra
        # Individual spectra
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
    
    # if plot:  # plot the correlation profiles
    #     # Correlations between peaks
    #     freqs_dev = freqs_blur - freqs_blur[len(freqs_blur) // 2]  # zero frequency at center of plot
    #     if num_peaks > 1:
    #         # fig = plt.figure(figsize=(7, 5))
    #         fig = plt.figure(figsize=(5.5, 4))
    #         for key, corr_vals in correlations.items():
    #             if 'off' in key:
    #                 continue
    #             plt.plot(freqs_dev, corr_vals / np.max(np.abs(corr_vals)), label=key)
    #         plt.ylabel('Normalized correlation')
    #         plt.xlabel(f'Deviation from autocorrelation peak ({freqs_blur.unit})')
    #         plt.title('Correlation between peaks')
    #         plt.legend()
    #         plt.tight_layout()
    #         # plt.savefig("plots/correlation_peaks", dpi=300)
    #         plt.show()

    return {
        'spectra':               spectra,
        'correlations':          correlations,
        'peak_times':            peak_times,
        'peak_boundaries':       peak_boundaries,
        'peak_power_idx':        peak_power_idx,
        'brightest_peak':        brightest_peak,
        'time_stream':           time_stream,
        'd0_blur':               d0_blur,
        'freqs_blur':            freqs_blur,
        'continuous_correlation': cont_corr,
        'off_pulse_region':      peak_boundaries[-1],
    }


# ── Helpers called by find_means ──────────────────────────────────────────

def _fit_autocorrelation(auto_corr, freqs_blur, n):
    """
    Fit parabola and triangular models to the central peak of
    an autocorrelation function.

    The ``n`` frequency samples nearest to the peak are used for
    fitting.  The peak sample itself is set to NaN before fitting
    (perfect self-correlation is not informative about the width).

    .. note::
        *Mutates* ``auto_corr`` in-place: the maximum sample is set to
        NaN and the array is normalised by the next-largest value.

    Parameters
    ----------
    auto_corr  : ndarray — modified in-place
    freqs_blur : Quantity
    n          : int

    Returns
    -------
    dict with keys:
        middle, central_freq, nearest_points,
        popt_parabola, unc_parabola,
        w_nom_parabola, w_nom_unc_parabola, mu_nom_parabola,
        popt_triangle, unc_triangle,
        w_nom_triangle, w_nom_unc_triangle, mu_nom_triangle,
        r2_parabola, r2_triangle
    """
    middle         = int(np.argmax(auto_corr))
    nearest_points = np.sort(
        np.argsort(np.abs(np.arange(len(auto_corr)) - middle))[:n]
    )

    # Normalise (in-place, preserving original behaviour)
    auto_corr[np.argmax(auto_corr)] = np.nan
    auto_corr /= np.nanmax(auto_corr)

    # # find width across at autocorrelation=0.5 to print
    # width_idx = np.where(auto_corr >= 0.5)[0]
    # if len(width_idx) > 1:
    #     width = freqs_blur[width_idx[-1]] - freqs_blur[width_idx[0]]
    # else:
    #     width = np.nan
    # print(f"Autocorrelation width at 0.5: {width:.3f}")

    x_data = freqs_blur[nearest_points].value
    y_data = auto_corr[nearest_points]
    p0     = [0.002, freqs_blur[middle].value, 1.2]

    popt_p, pcov_p = curve_fit(parabola,   x_data, y_data, nan_policy='omit', p0=p0)
    popt_t, pcov_t = curve_fit(triangular, x_data, y_data, nan_policy='omit', p0=p0)
    unc_p = np.sqrt(np.diag(pcov_p))
    unc_t = np.sqrt(np.diag(pcov_t))

    r2_p = _compute_r2(y_data, parabola(x_data, *popt_p),   use_nan=True)
    r2_t = _compute_r2(y_data, triangular(x_data, *popt_t), use_nan=True)

    return {
        'middle':          middle,
        'central_freq':    freqs_blur[middle],
        'nearest_points':  nearest_points,
        'popt_parabola':   popt_p,  'unc_parabola':   unc_p,
        'w_nom_parabola':  popt_p[0], 'w_nom_unc_parabola': unc_p[0], 'mu_nom_parabola': popt_p[1],
        'popt_triangle':   popt_t,  'unc_triangle':   unc_t,
        'w_nom_triangle':  popt_t[0], 'w_nom_unc_triangle': unc_t[0], 'mu_nom_triangle': popt_t[1],
        'r2_parabola':     r2_p,
        'r2_triangle':     r2_t,
    }


# ── More helpers called by find_means ──────────────────────────────────────────
def _select_cross_corr_fit_points(cross_corr, middle, n):
    """
    Select up to ``n`` fitting points centred on ``middle`` by expanding
    outward while the correlation remains above 0.5.

    Parameters
    ----------
    cross_corr : ndarray — normalised cross-correlation
    middle     : int     — starting index (from autocorrelation fit)
    n          : int     — maximum number of points

    Returns
    -------
    nearest_points : ndarray of int, sorted ascending
    """
    pts       = [middle - 1, middle, middle + 1]
    left_idx  = middle - 2
    right_idx = middle + 2
    lim = -1.0  # allow points with negative correlation, but not below this threshold

    while (left_idx >= 0 and right_idx < len(cross_corr)) and len(pts) < n:
        if cross_corr[left_idx] > lim:
            pts.append(left_idx);  left_idx  -= 1
        else:
            left_idx = -1                     # stop expanding left

        if cross_corr[right_idx] > lim:
            pts.append(right_idx); right_idx += 1
        else:
            right_idx = len(cross_corr)       # stop expanding right

    return np.sort(pts)


def _fit_cross_correlation(cross_corr, freqs_blur, nearest_points, auto_params, N=0.1):
    """
    Fit parabola and triangular models to a normalised cross-correlation.

    The width parameter ``w`` is constrained to ``w_nom ± N * w_nom_unc``
    (derived from the autocorrelation) and the centre ``mu`` to
    ``mu_nom ± 200`` (in the same frequency units).

    Parameters
    ----------
    cross_corr     : ndarray — already normalised by the caller
    freqs_blur     : Quantity
    nearest_points : ndarray of int
    auto_params    : dict returned by ``_fit_autocorrelation``
    N              : float — fractional width tolerance (default 0.1)

    Returns
    -------
    popt_p, unc_p, popt_t, unc_t : ndarrays of fit parameters / uncertainties
    r2_p, r2_t                   : float R² for each model
    """
    middle = auto_params['middle']
    x_data = freqs_blur[nearest_points].value
    y_data = cross_corr[nearest_points]

    def _bounds(w_nom, w_unc, mu_nom):
        lo = [w_nom - N * w_unc, mu_nom - 200, -np.inf]
        hi = [w_nom + N * w_unc, mu_nom + 200,  np.inf]
        if lo[0] == hi[0]:
            hi[0] += 1e-10
        return (lo, hi)

    bounds_p = _bounds(auto_params['w_nom_parabola'],
                       auto_params['w_nom_unc_parabola'],
                       auto_params['mu_nom_parabola'])
    try:
        popt_p, pcov_p = curve_fit(
            parabola, x_data, y_data,
            p0=[auto_params['w_nom_parabola'], freqs_blur[middle].value, np.nanmax(y_data)],
            bounds=bounds_p, nan_policy='omit'
        )
        unc_p = np.sqrt(np.diag(pcov_p))
        r2_p = _compute_r2(y_data, parabola(x_data, *popt_p), use_nan=True)
    except RuntimeError:
        popt_p = np.array([0, 0, 0])
        unc_p  = np.array([1, 1, 1])
        r2_p   = -100  # flag for failed fit

    bounds_t = _bounds(auto_params['w_nom_triangle'],
                       auto_params['w_nom_unc_triangle'],
                       auto_params['mu_nom_triangle'])
    try:
        popt_t, pcov_t = curve_fit(
            triangular, x_data, y_data,
            p0=[auto_params['w_nom_triangle'], freqs_blur[middle].value, np.nanmax(y_data)],
            bounds=bounds_t, nan_policy='omit'
        )
        unc_t = np.sqrt(np.diag(pcov_t))
        r2_t = _compute_r2(y_data, triangular(x_data, *popt_t), use_nan=True)
    except RuntimeError:
        popt_t = np.array([0, 0, 0])
        unc_t  = np.array([1, 1, 1])
        r2_t   = -100  # flag for failed fit

    return popt_p, unc_p, popt_t, unc_t, r2_p, r2_t


def find_means(correlations, ns_times, freqs_blur, n=11,
               plot=False, verbose=0, plot_triangle=False):
    """
    Fit correlation profiles to extract each nanoshot's central
    frequency and select the better-fitting model (parabola vs triangle).

    Operates in two passes over ``correlations``:

    1. **Autocorrelation pass** — fits the self-correlation of the
       brightest peak to determine the profile width and central
       frequency.  Mutates the autocorrelation array in-place.
    2. **Cross-correlation pass** — fits each peak-to-peak correlation
       using the width from pass 1 as a constrained prior.  Normalises
       each cross-correlation array in-place.

    Parameters
    ----------
    correlations : dict   — from ``correlate_nanoshots``; arrays mutated in-place
    ns_times     : dict   — {peak_index: Quantity time}
    freqs_blur   : Quantity
    n            : int    — number of frequency points used for fitting
    plot         : bool
    verbose      : int    — 0 = silent, 1 = model-choice summary, 2 = all params
    plot_triangle: bool   — if True, shows triangle fits in plots

    Returns
    -------
    mus          : dict  {key: (mu_value, mu_uncertainty)}
    ws           : dict  {key: (w_value,  w_uncertainty)}
    ns_times     : dict  (passed through unchanged)
    r2_avg       : float — average R² across all correlations
    central_freq : Quantity scalar — autocorrelation peak frequency
    plot_triangle: bool  — True if triangle gave better average R²
    """
    mus_parabola = {};  ws_parabola  = {}; As_parabola = {};  r2_avg_parabola = np.array([])
    mus_triangle = {};  ws_triangle  = {}; As_triangle = {};  r2_avg_triangle = np.array([])
    nearest_points_dict = {}
    auto_params  = None   # set by autocorrelation pass, reused by cross-corr pass

    if plot:
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    # ── Pass 1: autocorrelation ──────────────────────────────────────────────
    for key, corr_arr in correlations.items():
        if key[5] != key[7]:
            continue

        auto_params = _fit_autocorrelation(corr_arr, freqs_blur, n)

        As_parabola[key] = (auto_params['popt_parabola'][2], auto_params['unc_parabola'][2])
        As_triangle[key] = (auto_params['popt_triangle'][2], auto_params['unc_triangle'][2])
        mus_parabola[key] = (auto_params['popt_parabola'][1], auto_params['unc_parabola'][1])
        ws_parabola[key]  = (auto_params['popt_parabola'][0], auto_params['unc_parabola'][0])
        mus_triangle[key] = (auto_params['popt_triangle'][1], auto_params['unc_triangle'][1])
        ws_triangle[key]  = (auto_params['popt_triangle'][0], auto_params['unc_triangle'][0])
        r2_avg_parabola  += auto_params['r2_parabola']
        r2_avg_triangle  += auto_params['r2_triangle']
        nearest_points_dict[key] = auto_params['nearest_points']

        if verbose >= 2:
            p, u_ = auto_params['popt_parabola'], auto_params['unc_parabola']
            print(f"Autocorrelation parabola parameters for {key}: "
                  f"w = {p[0]:.3e} ± {u_[0]:.3e}, m = {p[1]:.3e} ± {u_[1]:.3e}")

        if plot:  # plot autocorrelation with fit points and fit curve
            central_freq     = auto_params['central_freq']
            freqs_dev        = (freqs_blur - central_freq).value
            nearest_pts      = auto_params['nearest_points']
            axs[0, 0].plot(freqs_dev, corr_arr)
            axs[0, 0].plot(freqs_dev[nearest_pts], corr_arr[nearest_pts],
                           'r', marker='o', linestyle='None', label='Fit points')
            if plot_triangle:
                fit_vals = triangular(freqs_blur[nearest_pts].value, *auto_params['popt_triangle'])
                mu_dev   = mus_triangle[key][0] - central_freq.value
                mu_unc   = mus_triangle[key][1]
                axs[0, 0].plot(freqs_dev[nearest_pts], fit_vals, 'g--', label='Triangle fit')
                axs[0, 0].axvline(mu_dev, color='g', linestyle=':', alpha=0.7)
                axs[0, 0].fill_betweenx([0, 1], mu_dev - mu_unc, mu_dev + mu_unc, color='g', alpha=0.2)
            else:
                fit_vals = parabola(freqs_blur[nearest_pts].value, *auto_params['popt_parabola'])
                mu_dev   = mus_parabola[key][0] - central_freq.value
                mu_unc   = mus_parabola[key][1]
                axs[0, 0].plot(freqs_dev[nearest_pts], fit_vals, 'g--', label='Parabola fit')
                axs[0, 0].axvline(mu_dev, color='g', linestyle=':', alpha=0.7)
                axs[0, 0].fill_betweenx([0, 1], mu_dev - mu_unc, mu_dev + mu_unc, color='g', alpha=0.2)
            axs[0, 0].set_xlabel(f'Deviation from autocorrelation peak ({central_freq.unit})')
            axs[0, 0].set_ylabel('Normalized Correlation')
            axs[0, 0].set_title(f'Auto-correlation for {key}')

    # ── Pass 2: cross-correlations ───────────────────────────────────────────
    plot_idx = 1
    for key, corr_arr in correlations.items():
        if key[5] == key[7] or 'off' in key:
            continue

        corr_arr /= np.nanmax(corr_arr)   # normalise in-place

        nearest_points = _select_cross_corr_fit_points(corr_arr, auto_params['middle'], n)
        nearest_points_dict[key] = nearest_points

        # set the max value of the nearest points to NaN to exclude from fit
        corr_arr[np.argmax(corr_arr[nearest_points]) + nearest_points[0]] = np.nan
        corr_arr /= np.nanmax(corr_arr[nearest_points])   # re-normalise after setting max to NaN
        
        popt_p, unc_p, popt_t, unc_t, r2_p, r2_t = _fit_cross_correlation(
            corr_arr, freqs_blur, nearest_points, auto_params
        )

        As_parabola[key] = (popt_p[2], unc_p[2])
        As_triangle[key] = (popt_t[2], unc_t[2])
        mus_parabola[key] = (popt_p[1], unc_p[1])
        ws_parabola[key]  = (popt_p[0], unc_p[0])
        mus_triangle[key] = (popt_t[1], unc_t[1])
        ws_triangle[key]  = (popt_t[0], unc_t[0])
        # r2_avg_parabola  += r2_p
        # r2_avg_triangle  += r2_t
        r2_avg_parabola = np.append(r2_avg_parabola, r2_p)
        r2_avg_triangle = np.append(r2_avg_triangle, r2_t)

        if verbose >= 2:
            print(f"Cross-correlation parabola parameters for {key}: "
                  f"w = {popt_p[0]:.3e} ± {unc_p[0]:.3e}, "
                  f"m = {popt_p[1]:.3e} ± {unc_p[1]:.3e}")

        if plot:  # plot cross-correlation with fit points and fit curve
            central_freq = auto_params['central_freq']
            freqs_dev    = (freqs_blur - central_freq).value
            ax = axs[plot_idx // 2, plot_idx % 2]
            ax.plot(freqs_dev, corr_arr)
            ax.plot(freqs_dev[nearest_points], corr_arr[nearest_points],
                    'r', marker='o', linestyle='None', label='Fit points')
            ax.axvline(0, color='r', linestyle='--', alpha=0.7)
            if plot_triangle:
                fit_vals = triangular(freqs_blur[nearest_points].value, *popt_t)
                mu_dev   = mus_triangle[key][0] - central_freq.value
                mu_unc   = mus_triangle[key][1]
                ax.plot(freqs_dev[nearest_points], fit_vals, 'g--', label='Triangle fit')
                ax.axvline(mu_dev, color='g', linestyle=':', alpha=0.7)
                ax.fill_betweenx([0, 1], mu_dev - mu_unc, mu_dev + mu_unc, color='g', alpha=0.2)
            else:
                fit_vals = parabola(freqs_blur[nearest_points].value, *popt_p)
                mu_dev   = mus_parabola[key][0] - central_freq.value
                mu_unc   = mus_parabola[key][1]
                ax.plot(freqs_dev[nearest_points], fit_vals, 'g--', label='Parabola fit')
                ax.axvline(mu_dev, color='g', linestyle=':', alpha=0.7)
                ax.fill_betweenx([0, 1], mu_dev - mu_unc, mu_dev + mu_unc, color='g', alpha=0.2)
            ax.set_xlabel(f'Deviation from autocorrelation peak ({central_freq.unit})')
            ax.set_ylabel('Normalized Correlation')
            ax.set_title(f'Cross-correlation for {key}')
            plot_idx += 1

    if plot:
        plt.tight_layout()
        plt.show()

    # r2_avg_parabola /= len(correlations)
    # r2_avg_triangle /= len(correlations)
    r2_avg_parabola = np.nanmean(r2_avg_parabola)
    r2_avg_triangle = np.nanmean(r2_avg_triangle)

    # if r2_avg_parabola < r2_avg_triangle:
    if False:
        if verbose >= 1:
            print(f"Triangle fit has higher average R2 ({r2_avg_triangle:.3f}) "
                  f"than parabola fit ({r2_avg_parabola:.3f}). Using triangle fit results.")
        return mus_triangle, ws_triangle, As_triangle, ns_times, r2_avg_triangle, auto_params['central_freq'], True, nearest_points_dict
    else:
        if verbose >= 1:
            print(f"Triangle fit has lower average R2 ({r2_avg_triangle:.3f}) "
                  f"than parabola fit ({r2_avg_parabola:.3f}). Using parabola fit results.")
        return mus_parabola, ws_parabola, As_parabola, ns_times, r2_avg_parabola, auto_params['central_freq'], False, nearest_points_dict


# ── Helpers called by get_correlation_peak_evolution ─────────────────────

def _read_and_prepare_data(file, freqs, is_pol, freq_start=144, freq_end=1488):
    """
    Open a baseband file, square the signal, optionally sum polarisations,
    and trim to the desired frequency range.

    Parameters
    ----------
    file       : baseband_tasks stream handle
    freqs      : Quantity — full frequency array
    is_pol     : bool — True if data has a polarisation axis to sum over
    freq_start : int — first frequency index to keep (inclusive)
    freq_end   : int — last frequency index to keep (exclusive)

    Returns
    -------
    data       : ndarray, shape (freq_end - freq_start, n_time)
    freqs_trim : Quantity
    """
    squared = Square(file)
    squared.seek(0)  # ensure we're at the start of the file
    raw     = squared.read()
    data    = (np.sum(raw, axis=1) if is_pol else raw).T
    norm_factor = np.median(data, axis=1, keepdims=True) / (1 - 2/36)**3
    data = data / norm_factor - 1
    # data = data / np.median(data, axis=1, keepdims=True)
    # data = data - np.mean(data, axis=1, keepdims=True)
    return data[freq_start:freq_end, :], freqs[freq_start:freq_end]


def _collect_sorted_results(mus, ns_times, central_freq):
    """
    Flatten the ``mus`` dict into parallel arrays of peak arrival times,
    frequency deviations from the autocorrelation centre, and fitting
    uncertainties, all sorted by ascending time.

    Parameters
    ----------
    mus          : dict  {key: (mu_value, mu_uncertainty)}
    ns_times     : dict  {peak_index: Quantity time}
    central_freq : Quantity scalar — autocorrelation peak frequency

    Returns
    -------
    peak_ts         : Quantity (us) — sorted peak arrival times
    freq_deviations : ndarray — fitted peak frequency minus autocorr peak
    uncertainties   : ndarray
    """
    peak_ts  = np.array([]) * u.ms
    ms_vals  = np.array([])
    ms_uncs  = np.array([])

    for key in mus.keys():
        ns_num   = int(key[7])
        peak_ts  = np.append(peak_ts,  ns_times[ns_num])
        ms_vals  = np.append(ms_vals,  mus[key][0])
        ms_uncs  = np.append(ms_uncs,  mus[key][1])

    order    = np.argsort(peak_ts)
    peak_ts  = peak_ts[order].to(u.us)
    ms_vals  = ms_vals[order]
    ms_uncs  = ms_uncs[order]

    return peak_ts, ms_vals - central_freq.value, ms_uncs


def _fit_and_plot_evolution(peak_ts, freq_deviations, uncertainties, central_freq, final_plot=True):
    """
    Fit a linear drift to the peak frequency deviations over time
    and plot the result with error bars.

    Parameters
    ----------
    peak_ts         : Quantity (us)
    freq_deviations : ndarray
    uncertainties   : ndarray
    central_freq    : Quantity scalar

    Returns
    -------
    popt_line : (slope, intercept)
    unc_line  : (sigma_slope, sigma_intercept)
    """
    popt_line, pcov_line = curve_fit(
        line, peak_ts.value, freq_deviations,
        sigma=uncertainties, absolute_sigma=True
    )
    unc_line = np.sqrt(np.diag(pcov_line))
    line_fit = line(peak_ts.value, *popt_line)
    if final_plot:
        # fig = plt.figure(figsize=(3.5, 3.))
        eb     = plt.errorbar(peak_ts, freq_deviations, yerr=uncertainties, fmt='o')
        colour = eb[0].get_color()
        plt.plot(peak_ts, line_fit, '--', c=colour,
                label=f'm = {popt_line[0]:.2f} ± {unc_line[0]:.2f}')
        plt.xlabel(rf"Time $({peak_ts.unit.to_string('latex')[1:-1]})$")
        plt.ylabel(f'Deviation from\nautocorr. peak ({central_freq.unit})')
        plt.title('Peak frequency deviation vs Time')
        plt.legend()
        plt.tight_layout()
        # plt.savefig("plots/bad_peak_frequency_evolution", dpi=300)
        plt.show()

    return popt_line, unc_line


def get_correlation_peak_evolution(data0, num_peaks, freqs, ts,
                                   blur=32, ns=[9, 11, 13], is_pol=True, avg_spectrum=None,
                                   plot=False, final_plot=True, 
                                   verbose=0):
    """
    Full analysis pipeline: read one nanoshot file, search for the best
    fitting parameter ``n``, extract the central frequency of each peak,
    and fit a linear drift across the pulse.

    The search loop tries each value in ``ns`` and accepts the first that
    achieves R² ≥ 0.9.  If none qualify, the ``n`` with the highest R²
    is used instead.

    Parameters
    ----------
    data0     : array-like
    num_peaks : int
    freqs     : Quantity — full frequency axis of the file
    ts        : Quantity — time axis (e.g. base_ts)
    blur      : int — frequency-averaging factor passed to correlate_nanoshots
    ns        : list of int — values of n to try for correlation fitting
    plot      : bool
    verbose   : int — 0 = silent, 1 = n-search progress, 2 = fit parameters
    is_pol    : bool — True if data has a polarisation axis

    Returns
    -------
    results   : [peak_ts, freq_deviations, uncertainties]
    popt_line : (slope, intercept)
    unc_line  : (sigma_slope, sigma_intercept)
    fit_metadata : dict with keys 'mus', 'ws', 'As', 'central_freq', 'plot_triangle', 'nearest_points_dict'
    """

    r2s           = np.zeros_like(ns, dtype=float)
    mus_dict      = {}
    ws_dict       = {}
    As_dict       = {}
    ns_times_dict = {}
    nearest_points_dict_dict = {}
    plot_triangle = False  # change to True if triangle fit is better for the best n
    final_n       = ns[0]

    if verbose >= 1:
        print("Trying different ns to find best fit for peak frequency evolution...")

    broke = False
    for i, n in enumerate(ns):
        if verbose >= 1:
            print(f"Testing n = {n}...")

        temp_results = correlate_nanoshots(
            data0, blur=blur, ts=ts, freqs=freqs, num_peaks=num_peaks, plot=False, avg_spectrum=avg_spectrum
        )
        mus, ws, As, ns_times, r2_avg, central_freq, temp_plot_triangle, nearest_pts_dict = find_means(
            temp_results['correlations'], temp_results['peak_times'],
            temp_results['freqs_blur'],  n=n, plot=False, verbose=verbose
        )

        if r2_avg >= 0.9:
            print(f"Using results for n = {n}")
            # Re-run with plot=plot to generate figures if requested.
            # Note: correlations have already been mutated by find_means above,
            # so this second find_means call operates on freshly generated ones.
            temp_results = correlate_nanoshots(data0, blur=blur, ts=ts, freqs=freqs,
                                num_peaks=num_peaks, plot=plot, avg_spectrum=avg_spectrum)
            mus, ws, As, ns_times, r2_avg, central_freq, temp_plot_triangle, nearest_pts_dict = find_means(
                temp_results['correlations'], temp_results['peak_times'],
                temp_results['freqs_blur'], n=n, plot=plot,
                verbose=0, plot_triangle=plot_triangle
            )
            broke = True
            break

        if r2_avg > np.nanmax(r2s):
            plot_triangle = temp_plot_triangle
            final_n       = n

        r2s[i]                       = r2_avg
        mus_dict[i]                  = mus
        ws_dict[i]                   = ws
        As_dict[i]                   = As
        ns_times_dict[i]             = ns_times
        nearest_points_dict_dict[i]  = nearest_pts_dict

    if not broke:
        if verbose >= 1:
            print("None of the ns gave a good fit (r2 >= 0.9). "
                  "Returning results for n with highest r2.")
        best_idx = int(np.argmax(r2s))
        print(f"Using results for n = {ns[best_idx]}")
        temp_results = correlate_nanoshots(
            data0, blur=blur, ts=ts, freqs=freqs, num_peaks=num_peaks, plot=plot, avg_spectrum=avg_spectrum
        )
        mus, ws, As, ns_times, r2_avg, central_freq, plot_triangle, nearest_pts_dict = find_means(
            temp_results['correlations'], temp_results['peak_times'],
            temp_results['freqs_blur'], n=final_n, plot=plot,
            verbose=0, plot_triangle=plot_triangle
        )

        mus      = mus_dict[best_idx]
        ws       = ws_dict[best_idx]
        As       = As_dict[best_idx]
        ns_times = ns_times_dict[best_idx]
        nearest_pts_dict = nearest_points_dict_dict[best_idx]

    peak_ts, freq_deviations, uncertainties = _collect_sorted_results(mus, ns_times, central_freq)

    popt_line, unc_line = _fit_and_plot_evolution(
        peak_ts, freq_deviations, uncertainties, central_freq, final_plot=final_plot
    )

    fit_metadata = {
        'mus': mus,
        'ws': ws,
        'As': As,
        'central_freq': central_freq,
        'plot_triangle': plot_triangle,
        'nearest_points_dict': nearest_pts_dict
    }

    return [peak_ts, freq_deviations, uncertainties], popt_line, unc_line, fit_metadata


# ── Function to create combined figure ─────────────────────────────────
def create_combined_figure(file, num_peaks, freqs, ts, blur=32, ns=[9, 11, 13], verbose=0, is_pol=True, f_idx=None, avg_spectrum=None):
    """
    Create a single large figure combining all analysis plots organized chronologically.
    
    Parameters
    ----------
    file : hdf5 file object
        Data file from correlate_nanoshots
    num_peaks : int
        Number of peaks in the data
    freqs : Quantity
        Frequency array with units
    ts : Quantity
        Time array with units
    blur : int
        Blurring factor (default 32)
    ns : list
        List of n values to try
    verbose : int
        Verbosity level
    is_pol : bool
        Whether data is polarized
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The combined figure object
    """
    
    # === EXTRACT DATA (plot=False to suppress individual plots) ===
    
    # Prepare raw data
    # data0, freqs_trim = _read_and_prepare_data(file, freqs, is_pol, freq_start=0, freq_end=len(freqs))
    # # data0, freqs_trim = _read_and_prepare_data(file, freqs, is_pol, freq_start=112, freq_end=1504)
    data0, freqs_trim = _read_and_prepare_data(file, freqs, is_pol, freq_start=144, freq_end=1488)
    # print(f"Data shape: {data0.shape}, Frequency range: {freqs_trim[0]} - {freqs_trim[-1]}")
    
    # Get nanoshot correlation data
    nanoshot_results = correlate_nanoshots(data0, blur=blur, ts=ts, freqs=freqs_trim, num_peaks=num_peaks, plot=False, avg_spectrum=avg_spectrum)
    
    d0 = data0
    d0_blur = nanoshot_results['d0_blur']
    time_stream = nanoshot_results['time_stream']
    peak_boundaries = nanoshot_results['peak_boundaries']
    freqs_blur = nanoshot_results['freqs_blur']
    spectra = nanoshot_results['spectra']
    correlations_full = nanoshot_results['correlations']
    peak_times_dict = nanoshot_results['peak_times']
    
    # Get peak evolution data
    results, popt_line, unc_line, fit_metadata = get_correlation_peak_evolution(data0, num_peaks, freqs=freqs_trim, ts=ts, blur=blur, avg_spectrum=avg_spectrum,
                                                                   ns=ns, plot=False, verbose=verbose, is_pol=is_pol, final_plot=False)
    ts_evolution, ms_values_deviation, ms_uncs = results
    
    mus = fit_metadata['mus']
    ws = fit_metadata['ws']
    As = fit_metadata['As']
    central_freq = fit_metadata['central_freq']
    plot_triangle = fit_metadata['plot_triangle']
    nearest_points_dict = fit_metadata['nearest_points_dict']
    
    # === FILTER DATA (exclude off-pulse) ===
    print(num_peaks, len(spectra))
    peak_spectra = spectra[:-1]  # Remove off-pulse spectrum
    correlations_peaks_only = {k: v for k, v in correlations_full.items() if "off" not in k}
    
    # === CREATE FIGURE ===
    fig = plt.figure(figsize=(12, 14))
    # gs_main = gridspec.GridSpec(4, 1, figure=fig, height_ratios=[0.20, 0.25, 0.35, 0.20], hspace=0.35)
    gs_main = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[0.30, 0.30, 0.40], hspace=0.35)
    
    # === ROW 1: Time stream + Peak Spectra ===
    # Create dynamic number of rows based on num_peaks
    gs_row1 = gridspec.GridSpecFromSubplotSpec(num_peaks, 2, subplot_spec=gs_main[0], 
                                              width_ratios=[0.6, 0.4], hspace=0.3)
    time_stream_xlim_fraction = 0.3  # Fraction of x-axis to show in time stream plot
    
    # Plot 1a: Dynamic Spectrum (spans top rows on left)
    center_idx = len(time_stream) // 2 + 30  # Shift center slightly to the right for better visibility 
    xlim_half = int(len(time_stream) * time_stream_xlim_fraction / 2)
    xmin = center_idx - xlim_half
    xmax = center_idx + xlim_half

    ax1a = fig.add_subplot(gs_row1[0:2, 0])
    im1 = ax1a.imshow(d0_blur, aspect='auto', origin='lower', cmap='magma', vmax = 0.85 * np.max(d0_blur),
                       extent=(ts[0].value, ts[-1].value, freqs_blur[0].value, freqs_blur[-1].value))
    ax1a.set_ylabel('Frequency')
    ax1a.tick_params(axis='x', which='both', labelbottom=False)
    # ax1a.set_title('Dynamic Spectrum')
    # show only the centre part of the time stream
    # ax1a.set_xlim(ts[xmin].value, ts[xmax].value)
    
    # Plot 1b: Time Stream with boundaries (spans bottom rows on left)
    if num_peaks > 2:
        ax1b = fig.add_subplot(gs_row1[2:, 0], sharex=ax1a)
    else:
        ax1b = fig.add_subplot(gs_row1[1, 0], sharex=ax1a)
    ax1b.plot(ts, time_stream)
    for boundary in peak_boundaries[:-1]:
        start, end = boundary
        ax1b.axvline(ts[start].value, color='r', linestyle='--', alpha=0.7)
        ax1b.axvline(ts[end].value, color='r', linestyle='--', alpha=0.7)
    ax1b.set_ylabel('Power')
    ax1b.set_xlabel('time [ms]')
    ax1b.set_xlim(ts[xmin].value, ts[xmax].value)

    # ax1b.set_title('Time Stream with Peak Boundaries')
    
    # Plot 2: Peak Spectra (one per row on right)
    for i, spectrum in enumerate(peak_spectra):
        ax = fig.add_subplot(gs_row1[i, 1])
        ax.plot(freqs_blur, spectrum, linewidth=1.5)
        # ax.tick_params(axis='y', which='both', labelleft=False)
        # ax.set_title(f'Spectrum of peak {i}')
        ax.set_ylabel('pk {}'.format(i), va='center')
        if i == len(peak_spectra) - 1:
            ax.set_xlabel(f'Frequency ({freqs_blur.unit})')
        else:
            ax.set_xticklabels([])

    # === ROW 2: Correlations between peaks and peak frequency evolution ===
    gs_row3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[1], 
                                              width_ratios=[0.5, 0.5], hspace=0)
    central_freq = freqs_blur[len(freqs_blur)//2]
    freqs_deviation = (freqs_blur - central_freq).value
    
    ax3 = fig.add_subplot(gs_row3[0])  # Center the correlation plot in the middle two columns
    for key, corr_vals in correlations_peaks_only.items():
        ax3.plot(freqs_deviation, corr_vals / np.max(np.abs(corr_vals)), label=key, linewidth=1.5)
    ax3.set_ylabel('Normalized correlation')
    ax3.set_xlabel(f'Deviation from Central Frequency ({freqs_blur.unit})')
    ax3.set_title("Correlation between peaks")
    ax3.legend(fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)

    ax_evo = fig.add_subplot(gs_row3[1])
    eb = ax_evo.errorbar(ts_evolution, ms_values_deviation, yerr=ms_uncs, fmt='o', linewidth=2, markersize=7)
    colour = eb[0].get_color()
    line_fit_vals = line(ts_evolution.value, *popt_line)
    ax_evo.plot(ts_evolution, line_fit_vals, '--', c=colour, linewidth=2.5, 
               label=f'm = {popt_line[0]:.2f} ± {unc_line[0]:.2f}')
    ax_evo.set_xlabel(f'Time ({ts_evolution.unit})')
    ax_evo.set_ylabel(f'Deviation from autocorr.\n peak ({freqs_blur.unit})')
    # ax_evo.set_title('Peak frequency deviation vs Time')
    ax_evo.legend(fontsize=10)

    ax_evo.grid(True, alpha=0.3)
    
    # === ROW 3: Correlation fits (2×2 grid) ===
    gs_row3 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_main[2], hspace=0.3, wspace=0.1)
    corr_xlim = 2000
    
    # Find auto and cross-correlation keys
    auto_corr_key = None
    cross_corr_keys = []
    for key in correlations_full.keys():
        if key[5] == key[7]:  # autocorrelation
            if auto_corr_key is None:
                auto_corr_key = key
        elif "off" not in key:
            cross_corr_keys.append(key)
    
    cross_corr_keys = cross_corr_keys[:3]  # Limit to 3 cross-correlations
    
    # Determine central frequency from autocorrelation
    if auto_corr_key:
        auto_corr_full = correlations_full[auto_corr_key].copy()
        middle = np.argmax(auto_corr_full)
        central_freq = freqs_blur[middle]
    else:
        central_freq = freqs_blur[len(freqs_blur)//2]
    
    freqs_deviation = (freqs_blur - central_freq).value
    
    # Plot auto-correlation (top-left cell only) with pre-computed fits
    ax_auto = fig.add_subplot(gs_row3[0, 0])
    if auto_corr_key:
        auto_corr = correlations_full[auto_corr_key].copy()
        auto_corr_peak_idx = np.argmax(auto_corr)
        auto_corr[auto_corr_peak_idx] = np.nan
        auto_corr /= np.nanmax(auto_corr)
        
        ymin = np.nanmin(auto_corr)
        ymax = np.nanmax(auto_corr)

        # Plot raw correlation
        ax_auto.plot(freqs_deviation, auto_corr, label=f'{auto_corr_key}', zorder=3)
        ax_auto.plot(freqs_deviation[nearest_points_dict[auto_corr_key]], auto_corr[nearest_points_dict[auto_corr_key]], c="r", marker='o', zorder=3)
        ax_auto.axvline(0, color='r', linestyle='--', alpha=0.7)  # central frequency is 0 in deviation space
        
        # Use pre-computed fit parameters from find_means()
        if auto_corr_key in mus:
            mu_val = mus[auto_corr_key][0]  # Peak frequency (absolute)
            w_val = ws[auto_corr_key][0]    # Width parameter
            A_val = As[auto_corr_key][0]    # Amplitude parameter
            unc_mu = mus[auto_corr_key][1]  # Uncertainty on peak frequency
            
            # Center peak position relative to central_freq for plotting
            mu_centered = mu_val - central_freq.value
            
            # Evaluate fit function using pre-computed parameters
            if plot_triangle:
                fit_curve = triangular(freqs_deviation, w_val, mu_centered, A_val)
                fit_label = 'Triangular fit'
            else:
                fit_curve = parabola(freqs_deviation, w_val, mu_centered, A_val)
                fit_label = 'Parabola fit'
            
            ax_auto.plot(freqs_deviation[nearest_points_dict[auto_corr_key]], fit_curve[nearest_points_dict[auto_corr_key]], 'g--', label=fit_label, zorder=4)
            
            # Plot central frequency with uncertainty shading
            ax_auto.axvline(mu_centered, ymin=np.min(auto_corr), ymax=np.max(auto_corr), color='g', linestyle=':', alpha=0.9, label='Fit peak', zorder=4)
            ax_auto.fill_betweenx([ymin, ymax], mu_centered - unc_mu, mu_centered + unc_mu, color='g', alpha=0.15, zorder=1)
        
        # ax_auto.set_xlabel(f'Deviation from peak ({central_freq.unit})')
        ax_auto.set_ylabel('Normalized Correlation')
        # ax_auto.set_title(f'Auto-correlation: {auto_corr_key}')
        ax_auto.grid(True, alpha=0.3)
        ax_auto.legend(fontsize=8)
        # ax_auto.set_ylim([-0.2, 1.2])
        # ax_auto.set_xlim([-corr_xlim, corr_xlim])
    
    # Plot cross-correlations (top-right, bottom-left, bottom-right) with pre-computed fits
    cross_positions = [(0, 1), (1, 0), (1, 1)]
    for i, key in enumerate(cross_corr_keys):
        if i < 3:
            row_idx, col_idx = cross_positions[i]
            ax = fig.add_subplot(gs_row3[row_idx, col_idx])
            cross_corr = correlations_full[key].copy()
            # set central peak to NaN to exclude from fit
            cross_corr[np.argmax(cross_corr[nearest_points_dict[key]]) + nearest_points_dict[key][0]] = np.nan
            cross_corr_max = np.nanmax(np.abs(cross_corr))
            cross_corr /= cross_corr_max
            
            # Plot raw correlation
            ymin = np.nanmin(cross_corr)
            ymax = np.nanmax(cross_corr)
            ax.plot(freqs_deviation, cross_corr, zorder=3, label=f'{key}')
            ax.plot(freqs_deviation[nearest_points_dict[key]], cross_corr[nearest_points_dict[key]], c="r", marker='o', zorder=3)
            ax.axvline(0, ymin=ymin, ymax=ymax, color='r', linestyle='--', alpha=0.7)  # central frequency is 0 in deviation space
            
            
            # Use pre-computed fit parameters from find_means()
            if key in mus:
                mu_val = mus[key][0]  # Peak frequency (absolute)
                w_val = ws[key][0]    # Width parameter
                A_val = As[key][0]    # Amplitude parameter
                unc_mu = mus[key][1]  # Uncertainty on peak frequency
                
                # Center peak position relative to central_freq for plotting
                mu_centered = mu_val - central_freq.value
                
                # Evaluate fit function using pre-computed parameters
                if plot_triangle:
                    fit_curve = triangular(freqs_deviation, w_val, mu_centered, A_val)
                    fit_label = 'Triangular fit'
                else:
                    fit_curve = parabola(freqs_deviation, w_val, mu_centered, A_val)
                    fit_label = 'Parabola fit'
                
                ax.plot(freqs_deviation[nearest_points_dict[key]], fit_curve[nearest_points_dict[key]], 'g--', zorder=4)
                
                # Central frequency marker with offset indication
                ax.axvline(mu_centered, ymin=ymin, ymax=ymax, color='g', linestyle=':', alpha=0.9, zorder=4)
                ax.fill_betweenx([ymin, ymax], mu_centered - unc_mu, mu_centered + unc_mu, color='g', alpha=0.15, zorder=1)
            
            if i != 0:
                ax.set_xlabel(f'Deviation from peak ({central_freq.unit})')
            # ax.set_ylabel('Correlation')
            # ax.set_title(f'{key}')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
            # ax.set_ylim([-0.3, 1.2])
            # ax.set_xlim([-corr_xlim, corr_xlim])

    fig.suptitle(f'GP {f_idx}/4 with blur={blur}', fontsize=14, fontweight='bold', y=0.9)    
    plt.tight_layout()
    return fig


def get_average_spectrum(files, blurs, base_ts=None, num_peaks_arr=None, blur_nom = 32):
    avg_spect = []
    for fn, file in enumerate(files):
        data, freqs  = _read_and_prepare_data(file, file.header0['frequency'], is_pol=True, freq_start=144, freq_end=1488)
        nom_freqs =  file.header0['frequency'][::blur_nom]
        _, d0_blur, time_stream, freqs_blur = _blur(data, blurs[fn], freqs)

        if num_peaks_arr is not None:
            peak_boundaries, peaks = find_peak_boundaries(time_stream, num_peaks_arr[fn])
            weights = time_stream[peaks] / np.max(time_stream[peaks])  # normalise weights to [0, 1]
            if base_ts is None:
                print("You must include base_ts if using num_peaks_arr to find peak boundaries.")
            peak_times  = {i: base_ts[pk] for i, pk in enumerate(peaks)}
            off_start   = peak_boundaries[-1][0]  # start of off-pulse region is end of last peak boundary
            off_end     = peak_boundaries[-1][1]  # end of off-pulse region is end of last peak boundary + pad

            spectra = _extract_and_normalize_spectra(d0_blur, peak_boundaries, off_start, off_end)
            spectra = spectra[:-1]  # remove the noise spectrum
            spectrum = np.average(spectra, axis=0, weights=weights)  # average over peaks
            # spectrum = np.mean(spectra, axis=0)  # average over peaks

        else:
            spectrum = tools.get_spectrum(d0_blur)
        
        spectrum = np.interp(nom_freqs, freqs_blur, spectrum)
        avg_spect.append(spectrum)
    avg_spect = np.mean(avg_spect, axis=0)
    return avg_spect - np.mean(avg_spect), nom_freqs