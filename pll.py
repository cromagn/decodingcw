from biquad_module import Biquad
from scipy.io import wavfile
from pylab import *
from math import *
import random, re


def ntrp(x, xa, xb, ya, yb):
    return (x - xa) * (yb - ya) / (xb - xa) + ya

#sample_rate, test_s = wavfile.read('C:/tmp/morse/PLL/E_Morse_1dit_1000_17.wav')
#sample_rate, test_s = wavfile.read('C:/tmp/morse/PLL/I_Morse_2dit_1000_17.wav')
#sample_rate, test_s = wavfile.read('C:/tmp/morse/PLL/S_Morse_3dit_1000_17.wav')
#sample_rate, test_s = wavfile.read('C:/tmp/morse/PLL/5_Morse_5dit_1000_17.wav')
#sample_rate, test_s = wavfile.read('C:/tmp/morse/PLL/H_Morse_4dit_1000_17.wav')
#sample_rate, test_s = wavfile.read('C:/tmp/morse/PLL/C_Morse_dot_dit_dot_dit_1000_17.wav')
sample_rate, test_s = wavfile.read('C:/tmp/morse/PLL/Vero_4.wav')
#sample_rate = 40000.0  # sampling frequency
print (sample_rate)
print (len(test_s))

cf = 1000

pll_integral = 0
old_ref = 0
pll_cf = 900 #1000
pll_loop_gain = 0.00003
pll_loop_gain = 0.28#0.009   0.1 0.09
ref_sig = 0

invsqr2 = 1.0 / sqrt(2.0)

cutoff = .06  # Units Hz
cutoff = 240  #7   110 80 60 180

loop_lowpass = Biquad(Biquad.LOWPASS, cutoff, sample_rate, invsqr2)

lock_lowpass = Biquad(Biquad.LOWPASS, cutoff, sample_rate, invsqr2)

ta = []
da = []
db = []
sn = []
xx = []

noise_level = 0 # +40 db 100

dur = 10  # very long run time
dur= len(test_s)/sample_rate
#for n in range(int(sample_rate) * dur):
for n in range(len(test_s)):
    t = n / sample_rate

    # BEGIN test signal block
    window = (0, 1)[(t > dur * 0 and t < dur * .26) or (t > dur * .34 and t < dur * .432) or (t > dur * .51 and t < dur * .77) or (t > dur * .855 and t < dur * .94)] #0.75
    #test_sig = sin(2 * pi * cf * t) * window
    noise = (random.random() * 2 - 1) * noise_level
    #test_sig += noise
    test_sig=(test_s[n]/2000) + noise
    # END test signal block

    # BEGIN PLL block
    pll_loop_control = test_sig * ref_sig * pll_loop_gain
    pll_loop_control = loop_lowpass(pll_loop_control)
    pll_integral += pll_loop_control / sample_rate
    ref_sig = sin(2 * pi * pll_cf * (t + pll_integral))
    quad_ref = (ref_sig - old_ref) * sample_rate / (2 * pi * pll_cf)
    old_ref = ref_sig
    pll_lock = lock_lowpass(-quad_ref * test_sig)
    # END PLL block
    xx.append(test_sig)
    if (n % 1 == 0):
        ta.append(t)
        #da.append(window)
        db.append(pll_lock * 2)
        sn.append(test_sig)

# ylim(-.5, 1.5)
wavfile.write('C:/tmp/morse/PLL/Out.wav',sample_rate, array(xx))
lPLL= 'PLL Response with cutoff = ' + str( cutoff) +' and loop gain = ' + str(pll_loop_gain)
suptitle(lPLL, fontsize=9)
#plot(ta, da, label='Signal present')
plot(ta, sn, label='Signal')
plot(ta, db, label='PLL response')


grid(True)
legend(loc='lower right')
setp(gca().get_legend().get_texts(), fontsize=9)
locs, labels = xticks()
setp(labels, fontsize=8)
locs, labels = yticks()
setp(labels, fontsize=8)

gcf().set_size_inches(5, 3.75)

name = re.sub('.*?(\w+).*', '\\1', sys.argv[0])
savefig(name + '.png')

show()
