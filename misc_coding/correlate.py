import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import os

from baseband_tasks.io import hdf5
from baseband_tasks.dm import DispersionMeasure
import tools.correlation as correlation


# set up stuff

data_path = "../data/MPs_zipped/"
files = os.listdir(data_path)
# files = [f for f in files if f.endswith('.h5')]
files = [os.path.join(data_path, f) for f in files]

# define sn to be the number after "SN-" and before ".hdf5" in filename (files)
sns = np.zeros_like(files, dtype=float)
times = np.zeros_like(files, dtype=float)
for i, file in enumerate(files):
    sn_start = file.find('SN-') + len('SN-')
    sn_end = file.find('.hdf5')
    sn = float(file[sn_start:sn_end])
    sns[i] = sn
    time_start = file.find('MJD-') + len('MJD-')
    time_end = file.find('_SN')
    time = float(file[time_start:time_end])
    times[i] = time

fs = np.array([hdf5.open(f, 'r') for f in files])

# great_ts = [60631.0849231004, 60631.095014371196, 60631.103663024645]
great_ts = np.array([60631.0849231004, 60631.09329920986, 60631.095014371196,
                     60631.10117956385, 60631.103663024645])
num_peaks_arr = np.array([4, 3, 3, 4, 3])
great_ts_idx = [np.argmin(np.abs(times - t)) for t in great_ts]
# print(great_ts_idx)

high_sns = np.where(sns > 20)[0]
# print(files)
fs = fs[great_ts_idx]

sample_f = fs[0]
# print(sample_f.sample_rate)
time_step = (1/sample_f.sample_rate).to(u.ms)
base_ts = np.linspace(0, sample_f.shape[0] * time_step, sample_f.shape[0]).to(u.ms)

blurs = [16, 32]

correlations = {}
freqs_blur = {}
peak_times = {}
fn = 0
f = fs[fn]

for fn, f in enumerate(fs):
    correlation.get_correlation_peak_evolution(f, num_peaks_arr[fn], f.header0['frequency'], base_ts, 
                                               blurs=blurs, plot=False, verbose=1)
plt.show()