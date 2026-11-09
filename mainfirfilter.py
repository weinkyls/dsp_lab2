# from mytask1.py
import numpy as np
import matplotlib.pyplot as plt
from firdesign import design_fir
from firfilter import FIRfilter

fs = 1000
ecg = np.loadtxt('Noisy_ECG_1000Hz_36.dat')
N = len(ecg)
t = np.arange(N) / fs

# 绘制时域图
plt.figure(figsize=(10, 4))
plt.plot(t, ecg, linewidth=0.6)
plt.title("Noisy ECG - time domain")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.savefig("task1_time.svg")

# 绘制频域图
X = np.abs(np.fft.fft(ecg))/ N
f = np.arange(N) * fs / N
plt.figure(figsize=(15, 4))
plt.plot(f[:N//2], X[:N//2], linewidth=0.6)
plt.title("Noisy ECG - frequency domain")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.savefig("task1_freq.svg")
plt.show()

fs = 1000
h = design_fir(fs, [[0, 0.5], [49, 51]])
M = len(h)
print("Number of coefficients:", M)

# plt.figure(figsize=(10, 4))
# plt.plot(h, linewidth=0.6)
# plt.title("FIR coefficients (impulse response)")
# plt.xlabel("Tap index n")
# plt.ylabel("h[n]")
# plt.savefig("task2_coefficients.svg")

# check the design: response should be 0 at DC and 50 Hz, 1 elsewhere
H = np.abs(np.fft.fft(h))
f = np.arange(M) * fs / M
plt.figure(figsize=(10, 4))
plt.plot(f[:M // 2], H[:M // 2])
plt.xlim(0, 100)
plt.title("Frequency response of the FIR filter")
plt.xlabel("Frequency (Hz)")
plt.ylabel("|H(f)|")
plt.savefig("task2_response.svg")

plt.show()

# for task 3
zoom_start = 5.0
zoom_duration = 1.0

ecg_filter = FIRfilter(h)
ecg_filtered = np.zeros(N)
for sample_index in range(N):
    ecg_filtered[sample_index] = ecg_filter.dofilter(ecg[sample_index])

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, ecg, linewidth=0.6)
plt.title("Raw ECG")
plt.ylabel("Amplitude")
plt.subplot(2, 1, 2)
plt.plot(t, ecg_filtered, linewidth=0.6)
plt.title("FIR filtered ECG")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig("task3_filtered.svg")

zoom_start_index = int(zoom_start * fs)
zoom_end_index = zoom_start_index + int(zoom_duration * fs)
plt.figure(figsize=(10, 4))
plt.plot(t[zoom_start_index:zoom_end_index], ecg_filtered[zoom_start_index:zoom_end_index])
plt.title("One heartbeat after FIR filtering")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.savefig("task3_heartbeat.svg")
plt.show()