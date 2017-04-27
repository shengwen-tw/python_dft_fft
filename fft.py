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

def fft(x):
    N = len(x)
    half_N = N // 2
    X = np.zeros(N, dtype=complex)

    if N == 1:
        X[0] = x[0]
    else:
        x_even = x[0::2] #get even until last element
        x_odd = x[1::2]  #get odd until last element
        X_even = fft(x_even)
        X_odd = fft(x_odd)
        for m in range(N):
            X[m] = X_even[m % half_N] + X_odd[m % half_N] * cmath.exp(-1j*2*math.pi*m/N)

    return X

def inv_fft(X):
    N = len(X)
    half_N = N // 2
    x = np.zeros(N, dtype=complex)

    if N == 1:
        x[0] = X[0]
    else:
        X_even = X[0::2]
        X_odd = X[1::2]
        x_even = inv_fft(X_even)
        x_odd = inv_fft(X_odd)
        for n in range(N):
            x[n] = x_even[n % half_N] + x_odd[n % half_N] * cmath.exp(1j*2*math.pi*n/N)

    return x

x = np.array([0, 5, 7, 3, 1, 0, 7, 2])

N = 2048

#Generate sin waves
m = 1
x1 = np.array([cmath.exp(-1j*2*math.pi/N*m*n) for n in range(N)])

X1 = fft(x1.real)
x1_reconst = inv_fft(X1) / len(X1)
plt.plot(x1)
plt.plot(X1 / len(X1))
plt.plot(x1_reconst)

plt.show()
