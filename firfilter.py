import numpy as np

# FIR class for part 3
class FIRfilter:
    def __init__(self,coefficients):
        self.ntaps=len(coefficients)
        self.h_reversed = np.asarray(coefficients)[::-1]
        self.buffer=np.zeros(self.ntaps)
        self.write_index = 0

    def dofilter(self,v):
        self.buffer[self.write_index] = v
        self.write_index = (self.write_index+1) % self.ntaps # update index

        oldest_len = self.ntaps - self.write_index
        y = np.dot(self.h_reversed[:oldest_len], self.buffer[self.write_index:]) + np.dot(self.h_reversed[oldest_len:], self.buffer[:self.write_index])
        return y