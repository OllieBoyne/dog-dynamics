"""Post/pre processing stage between image processing and ID,
and after ID as well"""
from scipy import signal
from matplotlib import pyplot as plt
import numpy as np

def smooth_data(data, freq=50, freq_cutoff_factor=8):
    """Smooths the joint data according to a butter lowpass filter."""

    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    coeff = butter_lowpass(freq / freq_cutoff_factor, freq)

    return signal.lfilter(*coeff, data.copy(), axis=0)

def low_pass_filter(x, T, freq_cutoff=0.0):
    """Simple low pass filter, works by fft.
    T = sampling time
    freq_cutoff is in Hz"""
    y = np.fft.fft(x, axis=0)
    cutoff_index = int(freq_cutoff * T)
    y[cutoff_index:] *= 0

    return np.abs(np.fft.ifft(y, axis=0))

def view_fft(x, T, cutoff=None):

    fs = x.size / T
    print(x.size, T)

    y = np.abs(np.fft.rfft(x)) / (2 * x.size) # 2*N is normalisation factor

    fig, (axt,axf,axfilt) = plt.subplots(nrows=3)
    axt.plot(x)
    frange = np.arange(0, fs/2 + 1e-5, 1/T)
    axf.plot(frange, y)

    if cutoff is not None:
        axfilt.plot(low_pass_filter(x, T, cutoff))

    plt.show()