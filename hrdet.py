import numpy as np
import matplotlib.pyplot as plt
from firdesign import design_fir
from firfilter import FIRfilter

fs = 1000
bands = [[0, 0.5], [49, 51]]

def prefilter(ecg):
    # remove baseline wander and 50 Hz, sample by sample (Task 3 filter)
    f = FIRfilter(design_fir(fs, bands))
    return np.array([f.dofilter(v) for v in ecg])

# Template: one heartbeat from another team's ECG (team 35)
ecg35 = prefilter(np.loadtxt('Noisy_ECG_1000Hz_35.dat'))
r35 = 6707                      # R peak of one clean heartbeat (found from plot)
pre, post = int(0.2 * fs), int(0.4 * fs)   # 600 ms: one P-QRS-T, shorter than one RR interval
template = ecg35[r35 - pre:r35 + post]
template = template - np.mean(template)   # make the template DC free

# Our ECG (team 36), pre-filtered in the same way
ecg = prefilter(np.loadtxt('Noisy_ECG_1000Hz_36.dat'))
N = len(ecg)
t = np.arange(N) / fs

# Matched filter = FIR filter with the time-reversed template
matched = FIRfilter(template[::-1])
threshold = 50                  # chosen from the plot of det
det = np.zeros(N)
r_peaks = []
peak = 0                        # index of the largest value above threshold
for n in range(N):
    y = matched.dofilter(ecg[n])
    det[n] = y * y              # squaring improves signal to noise ratio
    if det[n] > threshold:
        if det[n] > det[peak]:
            peak = n
    elif peak > 0 and det[n - 1] > threshold:
        r_peaks.append(peak)    # pulse has ended: its maximum is the detection
        peak = 0

# The matched filter output peaks 'post' samples after the R peak
r_peaks = np.array(r_peaks)
r_ecg = r_peaks - post

# Momentary heart rate = inverse of R-R interval
r_times = r_ecg / fs
rr = np.diff(r_times)
hr = 60 / rr                    # beats per minute
t_hr = r_times[1:]

# Heuristic: a heart rate must lie between 30 and 200 bpm
valid = (hr > 30) & (hr < 200)
print("Detected R peaks:", len(r_peaks), " rejected heart rates:", np.sum(~valid))

plt.figure(figsize=(8, 3))
plt.plot(np.arange(-pre, post) / fs * 1000, template)
plt.title("Matched filter template (one heartbeat from team 35)")
plt.xlabel("Time relative to R peak (ms)")
plt.ylabel("Amplitude (a.u.)")
for name, ms in [("P", -150), ("QRS", 0), ("T", 180)]:
    i = pre + int(ms / 1000 * fs)
    plt.annotate(name, (ms, template[i]), xytext=(ms, template[i] + 0.15), ha='center',
                 arrowprops=dict(arrowstyle='->'))
plt.ylim(-0.2, 1)
plt.tight_layout()
plt.savefig("task5_template.svg")

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, det, linewidth=0.6)
plt.axhline(threshold, color='r', linestyle='--', label=f'Threshold = {threshold}')
plt.plot(r_peaks / fs, det[r_peaks], 'rx', label='Detections')
plt.title("Squared matched filter output")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude$^2$ (a.u.)")
plt.ylim(0, 170)
plt.legend(loc='upper right')
plt.subplot(2, 1, 2)
plt.plot(t, ecg, linewidth=0.6)
plt.plot(t[r_ecg], ecg[r_ecg], 'rx', label='R peaks (shifted back by matched filter delay)')
plt.title("Filtered ECG with detected R peaks")
plt.ylim(-0.3, 1.1)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (a.u.)")
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("task5_detection.svg")

plt.figure(figsize=(10, 4))
plt.plot(t_hr[valid], hr[valid], 'o-')
plt.title("Momentary heart rate")
plt.xlabel("Time (s)")
plt.ylabel("Heart rate (bpm)")
plt.tight_layout()
plt.savefig("task5_heartrate.svg")
plt.show()
