import pickle
import numpy as np
import matplotlib.pyplot as plt

data = pickle.load(open('data_bandpass_test.p', 'rb'))

plt.figure()
plt.subplot(2,1,1)
plt.title(str(data['freq_low']) + ' Hz')
plt.plot(data['g_sub'], 20*np.log10(np.maximum(0.00000000001,data['peaks_low']-1)), label='Magnitude')
plt.axhline(y=20*np.log10(1/np.sqrt(2)), color='black', label='Cutoff Threshold')
plt.legend()
plt.ylabel('dB')

plt.subplot(2,1,2)
plt.title(str(data['freq_high']) + ' Hz')
plt.plot(data['g_sub'], 20*np.log10(np.maximum(0.00000000001,data['peaks_high']-1)), label='Magnitude')
plt.axhline(y=20*np.log10(1/np.sqrt(2)), color='black', label='Cutoff Threshold')
plt.legend()
plt.ylabel('dB')
plt.xlabel('g_sub (uS)')

plt.show()
