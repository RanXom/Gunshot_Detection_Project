import numpy as np
from scipy.signal import correlate

def cross_correlation(signal1, signal2):
    corr = correlate(signal1, signal2, mode='full')
    lag = np.argmax(corr) - len(signal1)
    return lag

def tdoa(signal1, signal2, sr):
    lag = cross_correlation(signal1, signal2)
    time_delay = lag / sr
    return time_delay
