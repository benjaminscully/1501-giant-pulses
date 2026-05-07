from tabnanny import verbose

import numpy as np
import matplotlib.pyplot as plt

from . import correlation
from . import tools

def get_corr(spectrum_a, spectrum_b, mean_a, mean_b, std_a, std_b):
    """
    Get the correlation between two spectra.
    """
    ones = np.ones_like(spectrum_a)
    ones_corr = np.correlate(ones, ones, mode='same')
    # A = np.sum((spectrum_a - mean_a) * (spectrum_b - mean_b)) / (len(spectrum_a) - 1)
    A = np.correlate(spectrum_a - mean_a, spectrum_b - mean_b, mode='same') / ones_corr
    # A = np.sum((spectrum_a) * (spectrum_b)) / (len(spectrum_a) - 1)
    A = A / (std_a * std_b)  # normalize by the product of the standard deviations to get thecorrelation coefficient
    B = mean_a * mean_b / ((mean_a - 1) * (mean_b - 1))
    return A * B

def correlate_two_MPs(file_A, file_B, blur_A, blur_B, plot=False, verbosity=0, avg_spect=None):
    """
    Correlate two different MPs. For each MP, choose only the brightest peak.
    """

    blur = np.minimum(blur_A, blur_B)
    if verbosity > 0:
        print(f"Using blur of {blur} for both MPs")

    data_a, freqs = correlation._read_and_prepare_data(file_A, file_A.header0['frequency'],
                                                        is_pol=True, freq_start=144, freq_end=1488)
    data_b, freqs = correlation._read_and_prepare_data(file_B, file_B.header0['frequency'],
                                                        is_pol=True, freq_start=144, freq_end=1488)

    _, da_blur, time_stream_a, freqs_blur_a = correlation._blur(data_a, blur, freqs=freqs)
    _, db_blur, time_stream_b, freqs_blur_b = correlation._blur(data_b, blur, freqs=freqs)

    peak_a = np.argmax(time_stream_a)
    peak_b = np.argmax(time_stream_b)

    spectrum_a = tools.get_spectrum(da_blur[:, peak_a-1:peak_a+2])
    spectrum_b = tools.get_spectrum(db_blur[:, peak_b-1:peak_b+2]) 

    mean_a = np.mean(spectrum_a)
    mean_b = np.mean(spectrum_b)
    std_a = np.std(spectrum_a)
    std_b = np.std(spectrum_b) 
    
    spectrum_a = (spectrum_a - mean_a)# / std_a
    spectrum_b = (spectrum_b - mean_b)# / std_b

    correlation_result = get_corr(spectrum_a, spectrum_b, mean_a, mean_b, std_a, std_b)

    if avg_spect is not None:
        avg_spectrum, avg_freqs = avg_spect
        # Interpolate the average spectrum to the frequencies of the current spectra
        avg_spectrum_interp = np.interp(freqs_blur_a, avg_freqs, avg_spectrum)
        # Subtract the average spectrum from the current spectra to decorrelate
        spectrum_a /= avg_spectrum_interp
        spectrum_b /= avg_spectrum_interp


    
    weight = (mean_a - 1)**2 * (mean_b - 1)**2 / (mean_a**2 * mean_b**2)

    other_correlation = np.correlate(spectrum_a, spectrum_b, mode='same')

    if plot:
        plt.plot(freqs_blur_a, spectrum_a, label='File A')
        plt.plot(freqs_blur_b, spectrum_b, label='File B')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Power (arb. units)')
        plt.legend()
        plt.show()

    if plot:
        cont_correlation = np.correlate(spectrum_a, spectrum_b, mode='same')
        plt.plot(freqs_blur_a - freqs_blur_a[len(freqs_blur_a)//2], cont_correlation)
        plt.xlabel('Lag')
        plt.ylabel('Correlation')
        plt.title('Cross-correlation of spectra')
        plt.show()

    return correlation_result, other_correlation, weight, freqs_blur_a