import numpy as np
import matplotlib.pyplot as plt

def design_fir(fs, bands):
    # resolution must fit the narrowest band and the lowest band edge
    df = min([f2 - f1 for f1, f2 in bands] + [f1 for f1, f2 in bands if f1 > 0])
    M = 2 * int(np.ceil(fs / df / 2))  # even, so the halves mirror

    X = np.ones(M)
    for f1, f2 in bands:
        X[int(f1 / fs * M):int(f2 / fs * M) + 1] = 0

    X[M // 2 + 1:] = X[1:M // 2][::-1]  # mirror, DC has no partner

    h = np.real(np.fft.ifft(X))
    h = np.concatenate((h[M // 2:], h[:M // 2]))  # make causal
    return h * np.hamming(M)