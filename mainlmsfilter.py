import numpy as np
import matplotlib.pyplot as plt
from firdesign import design_fir
from firfilter import FIRfilter

fs = 1000
ecg = np.loadtxt('Noisy_ECG_1000Hz_36.dat')
N = len(ecg)
t = np.arange(N) / fs

# Fixed FIR
h = design_fir(fs, [[0, 0.5], [49, 51]])
ecg_filter = FIRfilter(h)
ecg_filtered = np.zeros(N)

# LMS filter
ntaps = 100
learning_rate = 0.001
fnoise = 50
lms_filter = FIRfilter(np.zeros(ntaps))
ecg_lms = np.zeros(N)

for sample_index in range(N):
    ecg_filtered[sample_index] = ecg_filter.dofilter(ecg[sample_index])
    noise = np.sin(2.0 * np.pi * fnoise / fs * sample_index)
    ecg_lms[sample_index] = lms_filter.doFilterAdaptive(ecg[sample_index], noise, learning_rate)

# Time domain comparison
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, ecg_filtered, linewidth=0.6)
plt.title("FIR filtered ECG - time domain")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.subplot(2, 1, 2)
plt.plot(t, ecg_lms, linewidth=0.6)
plt.title("LMS filtered ECG - time domain")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig("task4_compare.svg")

# Single heartbeat comparison
delay = len(h) // 2
start = int(4.9 * fs)
end = int(5.5 * fs)
plt.figure(figsize=(10, 4))
plt.plot(t[start:end], ecg_filtered[start + delay:end + delay], label='Fixed FIR')
plt.plot(t[start:end], ecg_lms[start:end], label='LMS')
plt.title("One heartbeat - FIR and LMS comparison")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.savefig("task4_heartbeat.svg")

# Frequency domain comparison
X_fir = np.abs(np.fft.fft(ecg_filtered)) / N
X_lms = np.abs(np.fft.fft(ecg_lms)) / N
f = np.arange(N) * fs / N
k1 = int(45 / fs * N)
k2 = int(55 / fs * N) + 1
plt.figure(figsize=(10, 4))
plt.plot(f[k1:k2], X_fir[k1:k2], label='Fixed FIR')
plt.plot(f[k1:k2], X_lms[k1:k2], label='LMS')
plt.xlim(45, 55)
plt.title("FIR and LMS comparison - 45 to 55 Hz")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.tight_layout()
plt.savefig("task4_frequency.svg")
plt.show()
