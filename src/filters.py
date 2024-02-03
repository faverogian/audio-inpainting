from scipy.signal import butter, lfilter

def butter_lowpass(cutoff = 6000, fs = 22050, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff = 6000, fs = 22050, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y_filtered = lfilter(b, a, data)
    return y_filtered