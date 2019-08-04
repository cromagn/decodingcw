import struct
import matplotlib.pyplot as plt
import numpy as np
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
print("si parte")
#fs, data = wavfile.read('C:/tmp/morse/k5cm.wav')
fs, data = wavfile.read('C:/tmp/morse/wa6zty.wav')
print ("Sample freq: ",fs,"Num sample: ", len(data))
N=len(data)
T=1/fs
x = np.linspace(0.0, N*T, N)
# PLot orig data
plt.subplot(511)
plt.title("Original Signal Wave")
plt.plot(x,data)

# PLot orig sign spectrum
yf = fft(data)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
plt.subplot(512)
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
plt.subplot(514)
plt.title("Decoded LP signal")
plt.plot(x,ylp)
cutoff=100
ylp = butter_lowpass_filter(ysquared, cutoff, fs, order=4)
# Plot low pass
plt.subplot(515)
plt.title("Decoded LP signal")
plt.plot(x,ylp)
# PLot spectrum lowpass signal
# yf = fft(ylp)
# xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
# plt.subplot(616)
# plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
# plt.grid()
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
