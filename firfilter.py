import numpy as np

# FIR class for fixed and adaptive filtering
class FIRfilter:
    def __init__(self,coefficients):
        self.ntaps=len(coefficients)
        self.h_reversed = np.asarray(coefficients, dtype=float)[::-1].copy()
        self.buffer=np.zeros(self.ntaps)
        self.write_index = 0

    def dofilter(self,v):
        self.buffer[self.write_index] = v
        self.write_index = (self.write_index+1) % self.ntaps # update index

        oldest_len = self.ntaps - self.write_index
        y = np.dot(self.h_reversed[:oldest_len], self.buffer[self.write_index:]) + np.dot(self.h_reversed[oldest_len:], self.buffer[:self.write_index])
        return y

# Adaptive LMS filtering
    def doFilterAdaptive(self, signal, noise, learningRate):
        estimated_noise = self.dofilter(noise)
        error = signal - estimated_noise

        k = self.write_index
        for j in range(self.ntaps):
            self.h_reversed[j] = self.h_reversed[j] + error * learningRate * self.buffer[k]
            k += 1
            if k == self.ntaps:
                k = 0

        return float(error)
