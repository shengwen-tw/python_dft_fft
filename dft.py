#%matplotlib inline
import matplotlib.pyplot as plt
import pylab
import numpy as np
import math
import cmath

#input signal
x = np.array([0, 5, 7, 3, 1, 0, 7, 2])

def dft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)

    for m in range(N):
        for n in range(N):
            X[m] += x[n] * cmath.exp(-1j*2*math.pi/N*m*n)
    return X / math.sqrt(N)

def inv_dft(X):
    N = len(X)
    x = np.zeros(N, dtype=complex)
    for n in range(N):
        for m in range(N):
            x[n] += X[m] * cmath.exp(1j*2*math.pi*m*n/N)
        x[n] = x[n] / math.sqrt(N)

    return x

x = np.array([0, 5, 7, 3, 1, 0, 7, 2])

N = 300

#Generate sin waves
m = 15
x1 = np.array([cmath.exp(-1j*2*math.pi/N*m*n) for n in range(N)])

X1 = dft(x1.real)
x1_reconst = inv_dft(X1)
plt.plot(x1)
plt.plot(X1)
plt.plot(x1_reconst)

plt.show()
