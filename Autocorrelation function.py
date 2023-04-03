import numpy as np
import scipy as sp
from scipy import stats
from statsmodels.tsa.stattools import acf

# Autocorrelation function
def acw_f(ts, sr, fast=True):
    """
    Function computing the autocorrelation function
    and windows (50, 0, first nadir)
    ts = timeseries
    sr = sampling rate (in seconds)
    """
    sr = 1/sr # Sampling rate (s to Hz)
    acw = acf(ts, nlags=len(ts)-1, qstat=False, alpha=None, fft=fast)
    lags = np.arange(0, len(ts), 1)

    acw_50 = np.argmax(acw <= 0.5) / sr
    acw_0 = np.argmax(acw <= 0) / sr
    deriv = np.sign(np.diff(acw))
    acw_nadir = np.where(deriv == 1)[0][0] / sr
    return acw, lags, acw_50, acw_0, acw_nadir
