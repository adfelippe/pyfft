import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack
from scipy.signal import butter, lfilter, freqz
from scipy.signal import correlate
import dspytools as dsp



def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

fx = 12.34                                  # Frequency, in cycles per second, or Hertz
phase_x = 0 * np.pi / 180                   # Convert to degrees (rad x 180 / pi)
fy = 12.34
phase_y = 63 / 180 * np.pi                  # Convert to degrees (rad x 180 / pi)
fz = 0
n_samples = 256                             # Number of collected samples
sampling_time = 0.001                      # Sampling time in seconds
sampling_rate = 1 / sampling_time           # Sampling rate, or number of measurements per second (in Hz)
window = np.hanning(n_samples + 1)[:-1]     # Hanning window

t = np.linspace(0, n_samples * sampling_time, n_samples, endpoint=False)
x = 100 * np.sin(2 * np.pi * fx * t + phase_x) + 100 * np.sin(2 * np.pi * fx * 10 * t + phase_x)
y = 250 * np.sin(2 * np.pi * fy * t + phase_y)
# Add a DC offset
x = x + 150

# Remove DC offset from original signals before running FFT
x = dsp.dc_filter(x)
y = dsp.dc_filter(y)

# Values come in mg. Let's convert it to g
#x = x / 1000
#y = y / 1000

#x_filtered = dsp.butterworth_lowpass_filter(x, 30, sampling_rate, 4)
#y_filtered = dsp.butterworth_lowpass_filter(y, 30, sampling_rate, 4)
#print('Hanning window used...')
#x_filtered = x * window
#y_filtered = y * window
x_filtered = x
y_filtered = y
#z_filtered = z

phase_xy = dsp.find_phase_shift(x_filtered, y_filtered, fx)
print('Calculated X-Y phase shift: {:.2f}'.format(phase_xy))



fig, ax = plt.subplots()
ax.plot(t, x_filtered, label='X')
ax.plot(t, y_filtered, label='Y')
#ax.plot(t, z_filtered, label='Z')
#ax.plot(t, y_filtered, label='Y')
#ax.plot(t, z, label='Z')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Signal amplitude');
ax.legend()
#plt.show()

#fig, ax = plt.subplots()
##ax.plot(t, x_filtered, label='X')
#ax.plot(t, y_filtered, label='Y')
##ax.plot(t, z, label='Z')
#ax.set_xlabel('Time [s]')
#ax.set_ylabel('Signal amplitude');
#ax.legend()
#plt.show()

#fig, ax = plt.subplots()
#ax.plot(y_filtered, x_filtered, label='Orbit')
##ax.plot(t, z, label='Z')
#ax.set_xlabel('Y')
#ax.set_ylabel('X');
#ax.legend()

# FFT
X = fftpack.fft(x_filtered)
Y = fftpack.fft(y_filtered)

x_angle = np.angle(X, deg=True)
y_angle = np.angle(Y, deg=True)
fft_phase_x = x_angle[X.argmax()]
fft_phase_y = y_angle[Y.argmax()]
print('X angle: {0:5.2f} degrees'.format(fft_phase_x))
print('Y angle: {0:5.2f} degrees'.format(fft_phase_y))
phase_shift_rad = fft_phase_x - fft_phase_y
phase_shift = phase_shift_rad * 180 / np.pi
print('X-Z phase shift: {0:5.2f}'.format(phase_shift))

freq_x = fftpack.fftfreq(len(x_filtered)) * sampling_rate
freq_y = fftpack.fftfreq(len(y_filtered)) * sampling_rate

center_fx = dsp.find_center_frequency(X, freq_x)
#center_fy = find_center_frequency(Y, freq_y)
#center_fx = fft_interpolation(X, freq_x)
#center_fy = fft_interpolation(Y, freq_y)
print("\r\nX Center Frequency: {0:5.2f} Hz".format(center_fx))
#print("Y Center Frequency by interpolation: {0:5.2f} Hz".format(center_fy))

# Empty complex arrays
X_broken = np.zeros(len(X), dtype=complex)
Y_broken = np.zeros(len(Y), dtype=complex)

# Temporary buffers to avoid losing data in the upcoming loop
X_tmp = X.copy()
Y_tmp = Y.copy()

# Define size of the partial FFT
partial_fft_size = 64
X_max_value = []
X_max_index = []
Y_max_value = []
Y_max_index = []

# Assemble partial (incomplete) FFT for both X and Y signals
for i in range(partial_fft_size):
    X_max_index.append(np.abs(X_tmp).argmax())
    x_index = X_max_index[i]
    X_max_value = np.abs(X_tmp[x_index])
    X_broken[x_index] = X_tmp[x_index]
    X_tmp[X_max_index[i]] = 0
    Y_max_index.append(np.abs(Y_tmp).argmax())
    y_index = Y_max_index[i]
    Y_max_value = np.abs(Y_tmp[y_index])
    Y_broken[y_index] = Y_tmp[y_index]
    Y_tmp[y_index] = 0

#print("X Fundamental phase: {0:5.2f} rad".format(np.angle(X[max_1_idx])))
#print("X Fundamental phase: {0:5.2f} degrees".format(np.angle(X[max_1_idx], deg=True)))

iX = fftpack.ifft(X)
ibrokenX = fftpack.ifft(X_broken)
iY = fftpack.ifft(Y)
ibrokenY = fftpack.ifft(Y_broken)

fig, ax = plt.subplots()
ax.plot(t, iX, label='(X) Inverse FFT')
ax.plot(t, ibrokenX, 'r--', label='(X) Broken Inverse FFT')
#ax.plot(t, z, label='Z')
ax.set_xlabel('Amplitude')
ax.set_ylabel('t');
ax.legend()

fig, ax = plt.subplots()
ax.plot(t, iY, label='(Y) Inverse FFT')
ax.plot(t, ibrokenY, 'r--', label='(Y) Broken Inverse FFT')
#ax.plot(t, z, label='Z')
ax.set_xlabel('Amplitude')
ax.set_ylabel('t');
ax.legend()



# Only half of the FFT is useful (the other half is the same, but negative)
#print("<<< X >>>")
#for i in range(int(len(x_filtered) / 2)):
#    print("[X][{0:3d}] {1:5.2f} Hz: {2:5.2f}".format(i, freq_x[i], np.abs(X[i])))
#
#print("<<< Y >>>")
#for i in range(int(len(y_filtered) / 2)):
#    print("[Y][{0:3d}] {1:5.2f} Hz: {2:5.2f}".format(i, freq_y[i], np.abs(Y[i])))

fig, ax = plt.subplots()
ax.stem(freq_x, np.abs(X), use_line_collection=True)
ax.stem(freq_y, np.abs(Y), linefmt='--', use_line_collection=True)
ax.set_xlabel('Frequency in Hertz [Hz]')
ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
ax.set_xlim(-sampling_rate / 2, sampling_rate / 2)
plt.show()
