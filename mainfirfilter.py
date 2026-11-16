import numpy as np
import matplotlib.pyplot as plt
from firdesign import design_fir
from firfilter import FIRfilter

fs = 1000
ecg = np.loadtxt('Noisy_ECG_1000Hz_36.dat')
N = len(ecg)
t = np.arange(N) / fs

# ---------- Task 1 ----------
# time domain plot
plt.figure(figsize=(10, 4))
plt.plot(t, ecg, linewidth=0.6)
plt.title("Noisy ECG - time domain")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (a.u.)")
plt.savefig("task1_time.svg")

# zoom: one heartbeat
plt.figure(figsize=(10, 4))
plt.plot(t, ecg, linewidth=0.8)
plt.xlim(2, 3.5)
plt.title("Noisy ECG - zoom on one heartbeat")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (a.u.)")
plt.annotate("R peak", xy=(2.32, 0.89), xytext=(2.5, 0.75), arrowprops=dict(arrowstyle="->"))  # annotation
plt.annotate("50 Hz interference hides P and T waves", xy=(2.7, 0.2), xytext=(2.6, 0.5), arrowprops=dict(arrowstyle="->"))  # annotation
plt.savefig("task1_time_zoom.svg")

# frequency domain plot
X = np.abs(np.fft.fft(ecg)) / N
f = np.arange(N) * fs / N
plt.figure(figsize=(15, 4))
plt.plot(f[:N//2], X[:N//2], linewidth=0.6)
plt.title("Noisy ECG - frequency domain")
plt.xlabel("Frequency (Hz)")
plt.ylabel("|X(f)| (a.u.)")
plt.annotate("50 Hz mains interference", xy=(50, 0.075), xytext=(100, 0.06), arrowprops=dict(arrowstyle="->"))  # annotation
plt.savefig("task1_freq.svg")

# zoom: low frequencies
plt.figure(figsize=(10, 4))
plt.plot(f[:N//2], X[:N//2], linewidth=0.8)
plt.xlim(0, 8)
plt.ylim(0, 0.03)
plt.axvline(0.5, color="r", linestyle="--", label="cut-off 0.5 Hz")  # annotation
plt.title("Spectrum - zoom 0 to 8 Hz")
plt.xlabel("Frequency (Hz)")
plt.ylabel("|X(f)| (a.u.)")
plt.annotate("baseline / DC", xy=(0.1, 0.012), xytext=(0.6, 0.02), arrowprops=dict(arrowstyle="->"))  # annotation
plt.annotate("heart rate 1.43 Hz", xy=(1.43, 0.027), xytext=(2.0, 0.026), arrowprops=dict(arrowstyle="->"))
plt.annotate("harmonics", xy=(2.87, 0.008), xytext=(3.5, 0.018), arrowprops=dict(arrowstyle="->"))
plt.legend()
plt.savefig("task1_freq_zoom.svg")

# ---------- Task 2 ----------
h = design_fir(fs, [[0, 0.5], [49, 51]])
M = len(h)
print("Number of coefficients:", M)

# check the design: response should be 0 at DC and 50 Hz, 1 elsewhere
H = np.abs(np.fft.fft(h))
f = np.arange(M) * fs / M
plt.figure(figsize=(10, 4))
plt.plot(f[:M // 2], H[:M // 2])
plt.xlim(0, 100)
plt.title("Frequency response of the FIR filter")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Gain |H(f)|")
plt.annotate("DC / baseline removed (0-0.5 Hz)", xy=(0.5, 0.2), xytext=(8, 0.4), arrowprops=dict(arrowstyle="->"))  # annotation
plt.annotate("50 Hz removed (49-51 Hz)", xy=(50, 0.05), xytext=(60, 0.4), arrowprops=dict(arrowstyle="->"))  # annotation
plt.savefig("task2_response.svg")

# for task 3
zoom_start = 5.0
zoom_duration = 1.0

ecg_filter = FIRfilter(h)
ecg_filtered = np.zeros(N)
for sample_index in range(N):
    ecg_filtered[sample_index] = ecg_filter.dofilter(ecg[sample_index])

# filtered ecg
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, ecg, linewidth=0.6)
plt.title("Raw ECG")
plt.ylabel("Amplitude (a.u.)")
plt.subplot(2, 1, 2)
plt.plot(t, ecg_filtered, linewidth=0.6)
plt.title("FIR filtered ECG")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (a.u.)")
plt.tight_layout()
plt.savefig("task3_filtered.svg")

zoom_start_index = int(zoom_start * fs)
zoom_end_index = zoom_start_index + int(zoom_duration * fs)

# single heartbeat
plt.figure(figsize=(10, 4))
plt.plot(t[zoom_start_index:zoom_end_index], ecg_filtered[zoom_start_index:zoom_end_index])
tick_spacing = 0.05
plt.xticks(np.arange(zoom_start, zoom_start + zoom_duration + tick_spacing / 2, tick_spacing))
plt.grid(True)
# PQRST
wave_labels = [("P (5.27)", 5.27, 0.15), ("Q (5.387)", 5.387, -0.12), ("R (5.427)", 5.427, 0.1), ("S (5.482)", 5.482, -0.12), ("T (5.61)", 5.61, 0.15)]
for wave_name, wave_time, label_offset in wave_labels:
    wave_index = int(wave_time * fs)
    wave_amplitude = ecg_filtered[wave_index]
    plt.annotate(wave_name, xy=(wave_time, wave_amplitude), xytext=(wave_time, wave_amplitude + label_offset), ha="center", arrowprops=dict(arrowstyle="->"))
plt.ylim(-0.3, 0.9)
plt.title("Filtered ECG: single heartbeat")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (a.u.)")
plt.tight_layout()
plt.savefig("task3_heartbeat.svg")

X_filtered = np.abs(np.fft.fft(ecg_filtered)) / N
f_ecg = np.arange(N) * fs / N
max_freq = 60

# freq spectrum before and after FIR filter
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(f_ecg[:N // 2], X[:N // 2], linewidth=0.6)
plt.xlim(0, max_freq)
spectrum_ylim = plt.ylim()
plt.title("Raw ECG - frequency domain")
plt.ylabel("|X(f)| (a.u.)")
plt.subplot(2, 1, 2)
plt.plot(f_ecg[:N // 2], X_filtered[:N // 2], linewidth=0.6)
plt.xlim(0, max_freq)
plt.ylim(spectrum_ylim)
plt.title("FIR filtered ECG - frequency domain")
plt.xlabel("Frequency (Hz)")
plt.ylabel("|X(f)| (a.u.)")
plt.tight_layout()
plt.savefig("task3_spectrum.svg")
plt.show()