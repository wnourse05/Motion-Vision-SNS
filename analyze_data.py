import pickle
import matplotlib.pyplot as plt
import numpy as np

# Lowpass
data = pickle.load(open('data_lowpass.p', 'rb'))

plt.figure()
# Magnitude Response
plt.subplot(2,1,1)
plt.plot(data["frequencies"], 20*np.log10(data['outputPeaks']), label='Magnitude Response')
plt.axhline(y=20*np.log10(1/np.sqrt(2)), color='black', label='Cutoff Threshold')
plt.axvline(x=data['cutoff'], color='orange', label='Cutoff Freq')
plt.xscale('log')
plt.ylabel('Magnitude (dB)')
plt.title('Magnitude Response')
plt.legend()

plt.subplot(2,1,2)
plt.plot(data['frequencies'], data['phaseDiff'], label='Phase Response')
plt.xlabel('f (Hz)')
plt.ylabel('Phase (deg)')
plt.xscale('log')
plt.title('Phase Response')

# Highpass

data = pickle.load(open('data_highpass.p', 'rb'))

plt.figure()
plt.subplot(2,1,1)
plt.plot(data["frequencies"], 20*np.log10(data['outputPeaks']), label='Magnitude Response')
plt.axhline(y=20*np.log10(1/np.sqrt(2)), color='black', label='Cutoff Threshold')
plt.xscale('log')
plt.ylabel('Magnitude (dB)')
plt.title('Magnitude Response')
plt.legend()

plt.subplot(2,1,2)
plt.plot(data['frequencies'], data['phaseDiff'], label='Phase Response')
plt.xlabel('f (Hz)')
plt.ylabel('Phase (deg)')
plt.xscale('log')
plt.title('Phase Response')

plt.show()
