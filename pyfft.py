import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack
from scipy.signal import butter, lfilter, freqz

# This function estimates the center frequency based on the energy level
# of the 1st and 2nd order FFT peaks
# Args: fft     ->  Original calculated FFT
#       bins    ->  The frequency bins
def find_center_frequency(fft, bins):
    fft_peaks_value = []
    fft_peaks_index = []
    fft_half_len = int(len(fft) / 2)
    fft_buf = fft[0:fft_half_len].copy()

    # Find first and second highest energy levels
    # and their corresponding indexes
    for i in range(2):
        fft_peaks_index.append(np.abs(fft_buf).argmax())
        index = fft_peaks_index[i]
        fft_peaks_value.append(np.abs(fft_buf[index]))
        # Clear to find next highest energy level
        fft_buf[index] = 0

    # Estimate the center frequency based on energy levels
    # First off, sum the energy levels
    energy_sum = fft_peaks_value[0] + fft_peaks_value[1]
    # Then find the energy factor of the highest peak
    e_factor = fft_peaks_value[0] / energy_sum
    # Now we must subtract highest and 2nd highest value
    # corresponding frequency to find a delta
    delta_f = bins[fft_peaks_index[0]] - bins[fft_peaks_index[1]]
    # We can now estimate the frequency using the following formula:
    # f_est = (delta_f * e_factor) + second_highest_energy_level
    f_est = (delta_f * e_factor) + bins[fft_peaks_index[1]]
    return f_est




def remove_dc_offset(signal):

    signal_mean = np.mean(signal)
    signal_offset = signal - signal_mean
    return signal_offset

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

fx = 20.00                                  # Frequency, in cycles per second, or Hertz
fy = 13.33
fz = 0
n_samples = 128                             # Number of collected samples
sampling_time = 0.020                       # Sampling time in seconds
sampling_rate = 1 / sampling_time           # Sampling rate, or number of measurements per second (in Hz)
window = np.hanning(n_samples + 1)[:-1]     # Hanning window

t = np.linspace(0, n_samples * sampling_time, n_samples, endpoint=False)
#x = (300 * np.sin(2 * np.pi * fx * t + 2)) + (0 * np.sin(2 * np.pi * fx * t * 22 + 2))
#y = (400 * np.sin(2 * np.pi * fy * t)) + (0 * np.sin(2 * np.pi * fy * t * 10))
x = (50 * np.sin(2 * np.pi * fx * t))# + (15 * np.sin(2 * np.pi * fx * 25 * t))
y = (20 * np.sin(2 * np.pi * fy * t + (np.pi)))# + (9 * np.sin(2 * np.pi * fy * 25 * t + (np.pi)))
#z = 100 * np.sin(2 * np.pi * fz * t)

#x = [-78, -167, 27, 42, -226, 144, 82, -246, 19, -35, -167, 31, 246, -214, 70, -3, -113, -93, 292, -74, 50, 257, -113, -50, 183, -82, -66, 253, -39, -15, 97, 3, -230, 160, -23, -89, 164, 35, -191, 50, 234, -101, 62, 105, -156, 31, 210, -74, 3, 175, -109, -70, 164, -66, -89, 203, -15, -175, 160, -89, -199, 183, -15, -195, 136, -74, -238, 3, -3, -203, 117, 113, -210, 35, 7, -183, -23, 152, -191, 31, 210, -195, 11, 3, -164, -164, 230, -125, 31, 214, -113, -85, 156, -144, -66, 144, -58, -238, 78, -66, -121, 156, 39, -183, -35, 117, -222, 35, 187, -195, 70, 82, -214, 19, 214, -113, -11, 187, -125, -35, 125, -89, -82, 187, -7, -105, 171, -31, -171, 160, -23, -171, 156, -23, -191, 11, 46, -250, 117, 58, -175, 35, 62, -160, -35, 50, -242, 46, 160, -121, -3, 62, -117, 23, 164, -85, -171, 140, -3, -132, 191, 78, -132, 50, -46, -250, 152, 101, -187, 58, 0, -222, -11, 19, -109, 3, 164, -156, -171, 253, -97, -82, 171, 35, -238, 183, -23, -195, 113, -93, -269, 144, -66, -93, 160, 89, -187, 93, -19, -226, 70, 242, -117, 27, 253, -164, 62, 39, -109, -54, 281, -85, 42, 234, -136, -132, 78, -164, -39, 207, -35, -175, 85, 132, -207, 50, -35, -125, 105, 210, -152, 58, 238, -195, 35, 113, -152, 27, 101, -117, 19, 117, -125, -113, 171, -74, -70, 121, -23, -246, 93, -93, -234, 195, -3, -191]

#y = [136, 273, -11, 31, 292, -203, -62, 203, 85, -85, 281, 136, -125, 429, -46, -191, 398, 144, -19, 382, 15, -85, 339, 101, -222, 253, 343, -19, 74, 136, -195, 207, 269, -261, 78, 363, 27, 50, 257, -242, 136, 296, -242, 62, 269, 121, -27, 171, -167, -117, 273, 125, -113, 421, 191, -117, 82, 89, -175, 335, 285, -101, 398, 132, -203, 171, 148, -23, 378, 187, -62, 199, 156, -285, 78, 355, 19, 105, 250, -218, 15, 250, -222, -46, 351, 136, -11, 277, -164, -105, 246, -50, -105, 425, 148, -7, 97, -3, -164, 308, 304, -199, 394, 296, -78, 343, 179, -128, 234, 324, -234, 46, 371, 128, -23, 175, -160, -109, 261, 82, -109, 410, 160, -125, 62, 101, -171, 265, 289, -78, 363, 117, -199, 148, 167, -7, 355, 195, -50, 167, 171, -273, 46, 191, 23, 109, 281, -3, 7, 277, -222, -58, 390, 109, 7, 308, -156, -125, 281, -54, -97, 425, 199, -7, 93, 11, -171, 308, 261, -210, 402, 308, -160, 82, 117, -199, 378, 281, -82, 195, 195, -35, 46, 164, -15, 164, 289, -19, 15, 253, 78, -39, 132, 62, -97, 246, 132, -109, 390, -15, -136, 320, 125, 23, 285, 58, -54, 285, 113, -230, 210, 347, -23, 105, 175, -187, 199, 257, -281, 179, 359, 15, 78, 175, -242, 113, 281, -93, 58, 246, 125, -7, 269, -31, -105, 46, 261, -140, 316, 339, -31, 210, 93, -109, 328, 191, -222, 167, 281, 93, 54, 222, -171, 117, 335, -203]

#z = [1082, 617, 1195, 890, 960, 843, 792, 1230, 976, 1031, 1011, 1066, 1078, 937, 1042, 859, 1019, 929, 937, 1031, 1042, 1121, 847, 917, 996, 847, 945, 894, 769, 898, 1164, 1148, 851, 906, 1042, 1250, 804, 878, 972, 1015, 1187, 878, 914, 933, 1093, 980, 953, 914, 980, 1117, 761, 1175, 984, 1027, 1113, 585, 1140, 996, 921, 785, 792, 1312, 953, 1011, 1039, 1105, 1089, 933, 1054, 820, 917, 968, 953, 1046, 914, 1035, 960, 980, 1015, 886, 886, 925, 839, 859, 1152, 1164, 847, 886, 1015, 1355, 832, 824, 1042, 945, 1214, 777, 886, 921, 1093, 1144, 867, 898, 968, 1082, 902, 1007, 933, 1070, 1070, 722, 1085, 980, 960, 781, 769, 1257, 980, 1007, 1054, 1164, 1128, 902, 1093, 792, 863, 925, 1007, 1031]

# Remove DC offset from original signals before running FFT
#x = remove_dc_offset(x)
#y = remove_dc_offset(y)

# Values come in mg. Let's convert it to g
#x = x / 1000
#y = y / 1000

#x_filtered = butter_lowpass_filter(x, 30, sampling_rate, 5)
#y_filtered = butter_lowpass_filter(y, 30, sampling_rate, 5)
#x_filtered = x * window
#y_filtered = y * window
x_filtered = x
y_filtered = y

fig, ax = plt.subplots()
ax.plot(t, x_filtered, label='X')
#ax.plot(t, y_filtered, label='Y')
#ax.plot(t, z, label='Z')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Signal amplitude');
ax.legend()
#plt.show()

fig, ax = plt.subplots()
#ax.plot(t, x_filtered, label='X')
ax.plot(t, y_filtered, label='Y')
#ax.plot(t, z, label='Z')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Signal amplitude');
ax.legend()
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

freq_x = fftpack.fftfreq(len(x_filtered)) * sampling_rate
freq_y = fftpack.fftfreq(len(y_filtered)) * sampling_rate

center_fx = find_center_frequency(X, freq_x)
center_fy = find_center_frequency(Y, freq_y)
print("\r\nX Center Frequency: {0:5.2f} Hz".format(center_fx))
print("Y Center Frequency: {0:5.2f} Hz".format(center_fy))

# Empty complex arrays
X_broken = np.zeros(len(X), dtype=complex)
Y_broken = np.zeros(len(Y), dtype=complex)

# Temporary buffers to avoid losing data in the upcoming loop
X_tmp = X.copy()
Y_tmp = Y.copy()

# Define size of the partial FFT
partial_fft_size = 16
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
#plt.show()
