import time
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, firwin, lfilter

class ZeroPhaseLowPassFilter:
    def __init__(self, sample_rate, cutoff_frequency, order):
        self.sample_rate = sample_rate
        self.cutoff_frequency = cutoff_frequency
        self.order = order
        self.b, self.a = self._design_filter()

    def _design_filter(self):
        nyquist = 0.5 * self.sample_rate
        normal_cutoff = self.cutoff_frequency / nyquist
        b, a = butter(self.order, normal_cutoff, btype='low', analog=False)
        return b, a

    def filter_data(self, data):

        return filtfilt(self.b, self.a, data)

class FIRLowPassFilter:
    def __init__(self, sample_rate, cutoff_frequency, numtaps=101):
        self.sample_rate = sample_rate
        self.cutoff_frequency = cutoff_frequency
        self.numtaps = numtaps
        self.coefficients = self._design_filter()

    def _design_filter(self):
        nyquist = 0.5 * self.sample_rate
        normal_cutoff = self.cutoff_frequency / nyquist

        return firwin(self.numtaps, normal_cutoff, window='hamming')

    def filter_data(self, data):
        return lfilter(self.coefficients, 1.0, data)

if __name__ == "__main__":

    # sample_rate = 1833.0
    # t = np.linspace(0, 6.5, int(sample_rate), endpoint=False)
    # signal = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 12 * t)
    # print(t.shape)

    csv_file_path = '/home/iff/drlenv/src/iffenv/src/sensor_data_0.csv'

    df_read = pd.read_csv(csv_file_path)
    # print(df_read.head())
    full_signal = df_read.to_numpy()


    print(full_signal.shape)

    sample_rate = full_signal.shape[0]

    signal = full_signal[:,1]
    t = full_signal[:,0]
    print(t)


    cutoff_frequency = 10.0

    lpf = ZeroPhaseLowPassFilter(sample_rate, cutoff_frequency)


    lpf1 = FIRLowPassFilter(sample_rate, cutoff_frequency)


    ct = time.time()
    filtered_signal = lpf.filter_data(signal)
    print('zero time', (time.time() - ct))

    ct2 = time.time()
    filtered_signal1 = lpf1.filter_data(signal)
    print((time.time() - ct2) * 1000)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(t, signal, label='Original Signal')
    plt.plot(t, filtered_signal, label='Filtered Signal (Zero Phase)', color='red', linewidth=2)
    plt.plot(t, filtered_signal1, label='Filtered Signal (FIR Phase)', color='green', linewidth=2)
    plt.legend()
    plt.title('Low-Pass Filter Example (Zero Phase)')
    plt.xlabel('Time [seconds]')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()