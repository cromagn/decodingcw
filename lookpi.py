import logging
import math
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sg
from scipy.io import wavfile
from scipy.fftpack import fft
from scipy.fft import fftshift
from scipy.signal import butter, sosfilt, sosfreqz,lfilter
fSignal=3000
fMin=400
fMax=400

tau = np.pi * 2
max_samples = 1000000
debug = True
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
    f = sp.fft.fft(d, len(a))
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
#fs, data = wavfile.read('C:/tmp/CW/satnogs_5_lp20024.wav')
#fs, data = wavfile.read('C:/tmp/CW/satnogs_5_lphp.wav')
#fs, data = wavfile.read('C:/tmp/CW/satnogs_sample.wav')
#fs, data = wavfile.read('C:/tmp/CW/Qui/morse_carlo_20.wav')
#fs, data = wavfile.read('C:/tmp/CW/Qui/one_414_1.wav')
fs, data = wavfile.read('C:/tmp/CW/Qui/UsaQuesto.wav')

decimateFactor=4
datafilt = butter_bandpass_filter(data, fSignal-fMin, fSignal+fMax, fs, order=5)  #2600 3500 300 500
ddec=sg.decimate(datafilt,decimateFactor)
# intorno alla frequenza che sto cercando
T = 1.0 / fs * decimateFactor
N=len(ddec)
x = np.linspace(0.0, N*T, N)
tSlice=0.01 # 0.02 per 20
nslice=int((fs / decimateFactor) * tSlice)
ngraph= math.ceil(pow( N/ nslice,1/2))
print ("Freq Sample: ",fs)
print ("Num sample: ", N)
print ("Num sample in slice: ",nslice)
print ("Decimator: ",decimateFactor)
print ("Num Image X",ngraph)
#fig, axs = plt.subplots(ngraph, ngraph)
xf = np.linspace(0.0, 1.0/(2.0*T), nslice//2)
print(xf)
print(np.where((xf >(fSignal-fMin)) & (xf<(fSignal+fMax))))
yfval=[]
yfvalmax=[]
for x in range(0, N-nslice-1, nslice):
    yf = fft(ddec[x:x+nslice])
    yfval.append(yf)
    yfvalmax.append(max(2.0/nslice * np.abs(yf[0:nslice//2])))
    # plt.subplot(614)
plt.plot(yfvalmax)
plt.show()
symbols=wpcr(yfvalmax)
bits=slice_bits(symbols)
print(list(bits))
plt.plot(bits,'ro-')
plt.show()

morseChar={'A':[0,1,0,1,1,1,0], 'B': [0,1,1,1,0,1,0,1,0,1,0],'C':[0,1,1,1,0,1,0,1,1,1,0,1,0], 'D': [0,1,1,1,0,1,0,1,0],
           'E':[0,1,0], 'F': [0,1,0,1,0,1,1,1,0,1,0],'G':[0,1,1,1,0,1,1,1,0,1,0], 'H': [0,1,0,1,0,1,0,1,0],
           'I':[0,1,0,1,0], 'J': [0,1,0,1,1,1,0,1,1,1,0,1,1,1,0],'K':[0,1,1,1,0,1,0,1,1,1,0], 'L': [0,1,0,1,1,1,0,1,0,1,0],
           'M':[0,1,1,1,0,1,1,1,0], 'N': [0,1,1,1,0,1,0],'O':[0,1,1,1,0,1,1,1,0,1,1,1,0], 'P': [0,1,0,1,1,1,0,1,1,1,0,1,0],
           'Q':[0,1,1,1,0,1,1,1,0,1,0,1,1,1,0], 'R': [0,1,0,1,1,1,0,1,0],'S':[0,1,0,1,0,1,0], 'T': [0,1,1,1,0],
           'U':[0,1,0,1,0,1,1,1,0], 'V': [0,1,0,1,0,1,0,1,1,1,0],'W':[0,1,0,1,1,1,0,1,1,1,0], 'X': [0,0,1,1,1,0,1,0,1,0,1,1,1,0,0],
           'Y':[0,1,1,1,0,1,0,1,1,1,0,1,1,1,0], '1': [0,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0],'2':[0,1,0,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0],
           '3': [0,1,0,1,0,1,0,1,1,1,0,1,1,1,0],'4':[0,1,0,1,0,1,0,1,0,1,1,1,0], '5': [0,1,0,1,0,1,0,1,0,1,0],'6':[0,1,1,1,0,1,0,1,0,1,0,1,0],
           '7': [0,1,1,1,0,1,1,1,0,1,0,1,0,1,0],'8': [0,1,1,1,0,1,1,1,0,1,1,1,0,1,0,1,0],'9': [0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,0],'0': [0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0],
           'SP': [0,0,0],'SPW': [0,0,0,0,0]}

lb = len(bits)
decoded = [0] * lb
for chartofind, representation in morseChar.items():

    le=len(representation)
    res=0
    buckets = [0] * lb

    for b in range(lb-le):
        for i in range(le):
            if (bits[b+i]==representation[i]):
                res=res+1
        buckets[b]=res
        res=0
    maxElement = np.amax(buckets)
    print(maxElement, le)
    if(maxElement==le):
        result = np.where(buckets == np.amax(buckets))
        for x in result:
            decoded[x[0]]=chartofind
print(decoded)
X =  [x for x in decoded if not x==0]
print(X)
idx=0
# for xx in range(0,ngraph):
#     for yy in range(0,ngraph):
#         if idx<len(yfval):
#             print("Max: ", max(2.0/nslice * np.abs(yfval[idx][0:nslice//2])))
#             #axs[xx,yy].plot(xf, 2.0/nslice * np.abs(yfval[idx][0:nslice//2]))
#             idx=idx+1
#plt.show()
#plt.savefig('C:/tmp/CW/Qui/foo.png', bbox_inches='tight',pad_inches=0)

# fig, axs = plt.subplots(2, 2)
# axs[0, 0].plot(x, y)
# axs[0, 0].set_title('Axis [0,0]')
# axs[0, 1].plot(x, y, 'tab:orange')
# axs[0, 1].set_title('Axis [0,1]')
# axs[1, 0].plot(x, -y, 'tab:green')
# axs[1, 0].set_title('Axis [1,0]')
# axs[1, 1].plot(x, -y, 'tab:red')
# axs[1, 1].set_title('Axis [1,1]')
#
# for ax in axs.flat:
#     ax.set(xlabel='x-label', ylabel='y-label')
