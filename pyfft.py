import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack

# Return the value in Hz for each FFT bin
def fft_bin_hz(num, fs, size):
    return (num * fs / size)

f1 = 13.33                               # Frequency, in cycles per second, or Hertz
f2 = 40.00
n_samples = 128                         # Number of collected samples
sampling_time = 0.008                   # Sampling time in seconds
sampling_rate = 1 / sampling_time       # Sampling rate, or number of measurements per second (in Hz)

t = np.linspace(0, n_samples * sampling_time, n_samples, endpoint=False)
wave_1 = np.sin(2 * np.pi * f1 * t)
wave_2 = 0.5 * np.sin(2 * np.pi * f2 * t)
x = wave_1 + wave_2

fig, ax = plt.subplots()
ax.plot(t, x)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Signal amplitude');
#plt.show()

# FFT
X = fftpack.fft(x)
freqs = fftpack.fftfreq(len(x)) * sampling_rate

# Only half of the FFT is useful (the other half is the same, but negative)
for i in range(int(len(x) / 2)):
    print("[{0:3d}] {1:5.2f} Hz: {2:5.2f}".format(i, freqs[i], np.abs(X[i])))


fig, ax = plt.subplots()

ax.stem(freqs, np.abs(X))
ax.set_xlabel('Frequency in Hertz [Hz]')
ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
ax.set_xlim(-sampling_rate / 2, sampling_rate / 2)
ax.set_ylim(-5, n_samples)
plt.show()
