import struct
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.fftpack import fft
from scipy import signal as sg
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, sosfreqz,lfilter

#fs, data = wavfile.read('C:/Users/ROMAGNOLIC/Downloads/satnogs.wav')

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y


tau = np.pi * 2
max_samples = 1000000
debug = True

# determine the clock frequency
# input: magnitude spectrum of clock signal (numpy array)
# output: FFT bin number of clock frequency
def find_clock_frequency(spectrum):
    maxima = sp.signal.argrelextrema(spectrum, np.greater_equal)[0]
    while maxima[0] < 2:
        maxima = maxima[1:]
    if maxima.any():
        threshold = max(spectrum[2:-1])*0.8
        indices_above_threshold = np.argwhere(spectrum[maxima] > threshold)
        return maxima[indices_above_threshold[0]]
    else:
        return 0

def midpoint(a):
    mean_a = np.mean(a)
    mean_a_greater = np.ma.masked_greater(a, mean_a)
    high = np.ma.median(mean_a_greater)
    mean_a_less_or_equal = np.ma.masked_array(a, ~mean_a_greater.mask)
    low = np.ma.median(mean_a_less_or_equal)
    return (high + low) / 2

# whole packet clock recovery
# input: real valued NRZ-like waveform (array, tuple, or list)
#        must have at least 2 samples per symbol
#        must have at least 2 symbol transitions
# output: list of symbols
def wpcr(a):
    if len(a) < 4:
        return []
    b = (a > midpoint(a)) * 1.0
    d = np.diff(b)**2
    if len(np.argwhere(d > 0)) < 2:
        return []
    f = sp.fft(d, len(a))
    p = find_clock_frequency(abs(f))
    if p == 0:
        return []
    cycles_per_sample = (p*1.0)/len(f)
    clock_phase = 0.5 + np.angle(f[p])/(tau)
    if clock_phase <= 0.5:
        clock_phase += 1
    symbols = []
    for i in range(len(a)):
        if clock_phase >= 1:
            clock_phase -= 1
            symbols.append(a[i])
        clock_phase += cycles_per_sample
    if debug:
        print("peak frequency index: %d / %d" % (p, len(f)))
        print("samples per symbol: %f" % (1.0/cycles_per_sample))
        print("clock cycles per sample: %f" % (cycles_per_sample))
        print("clock phase in cycles between 1st and 2nd samples: %f" % (clock_phase))
        print("clock phase in cycles at 1st sample: %f" % (clock_phase - cycles_per_sample/2))
        print("symbol count: %d" % (len(symbols)))
    return symbols

# convert soft symbols into bits (assuming binary symbols)
def slice_bits(symbols):
    symbols_average = np.average(symbols)
    bits = (symbols >= symbols_average)
    return np.array(bits, dtype=np.uint8)


print("si parte")
#fs, data = wavfile.read('C:/tmp/morse/k5cm.wav')
fs, data = wavfile.read('C:/tmp/morse/wa6zty.wav')
#fs, data = wavfile.read('C:/tmp/morse/w1aw.wav')
print ("Sample freq: ",fs,"Num sample: ", len(data))
N=len(data)
T=1/fs
x = np.linspace(0.0, N*T, N)
# PLot orig data
plt.subplot(611)
plt.title("Original Signal Wave")
plt.plot(x,data)

# PLot orig sign spectrum
yf = fft(data)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
plt.subplot(612)
plt.title("Original Signal FFT")
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()

# identify freq
narr=2.0/N * np.abs(yf[0:N//2])
fmaxidx= np.where(narr == np.amax(narr))
fmax=int (xf[fmaxidx[0]])
print ("Detected signal @ :", fmax)
# band pass filter around max spectrum
lowcut=fmax-50
highcut=fmax+50
yfiltered = butter_bandpass_filter(data, lowcut, highcut, fs, order=6)
# # PLot filtered signal
# plt.subplot(613)
# plt.plot(x,yfiltered)

# PLot spectrum filtered signal
yf = fft(yfiltered)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
# plt.subplot(614)
# plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
# plt.grid()

# Square the signal
ysquared=yfiltered*yfiltered
# PLot squared signal
#plt.subplot(815)
#plt.plot(x,ysquared)

# PLot spectrum squared signal
yf = fft(ysquared)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
#plt.subplot(816)
#plt.plot(xf, 2.0/N * np.abs(yf[0:N//2])))
#plt.grid()

# apply lowpass
cutoff=20
ylp = butter_lowpass_filter(ysquared, cutoff, fs, order=4)
# Plot low pass
plt.subplot(513)
plt.title("Decoded LP signal")
plt.plot(x,ylp)
cutoff=50
ylp = butter_lowpass_filter(ysquared, cutoff, fs, order=4)
# Plot low pass
plt.subplot(614)
plt.title("Decoded LP signal")
plt.plot(x,ylp)
cutoff=100
ylp = butter_lowpass_filter(ysquared, cutoff, fs, order=4)
# Plot low pass
plt.subplot(615)
plt.title("Decoded LP signal")
plt.plot(x,ylp)
# PLot spectrum lowpass signal
# yf = fft(ylp)
# xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
# plt.subplot(616)
# plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
# plt.grid()
plt.subplot(616)
plt.title("Decoded  signal")
symbols=wpcr(ylp)
bits=slice_bits(symbols)
print(list(bits))
plt.plot(bits,'ro-')
plt.tight_layout()
plt.show()


#
# Number of sample points
# N = 88200
# lowcut=850
# highcut=950
# # sample spacing
# fs=44100
# T = 1.0 / fs
# x = np.linspace(0.0, N*T, N)
# y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x) + 0.5*np.sin(880.0 * 2.0*np.pi*x)
# ysq=(1 + sg.square(2.0*np.pi*x))/2
# yy=ysq*y
# yfiltered = butter_bandpass_filter(yy, lowcut, highcut, fs, order=6)
# yfilteredsq=yfiltered*yfiltered
# cutoff=100
# ylp = butter_lowpass_filter(yfilteredsq, cutoff, fs, order=6)
#
# plt.plot(x,yy)
# plt.show()
#
# plt.plot(x,yfiltered)
# plt.show()
# plt.plot(x,yfilteredsq)
# plt.show()
# plt.plot(x,ylp)
# plt.show()
#
# yf = fft(yfiltered)
# xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
# plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
# plt.grid()
# plt.show()
# f=open('x.wav','wb')
# for i in y:
#     f.write(struct.pack('b',int(i)))
# f.close()

