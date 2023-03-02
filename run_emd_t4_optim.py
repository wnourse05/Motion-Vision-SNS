import numpy as np
import torch
import matplotlib.pyplot as plt

from utilities import cutoff_fastest, device, gen_gratings
from motion_vision_networks import gen_emd_on_mcmc

#                   Retina          L1                                  L2                              L3                  Mi1         Mi9             Tm1             Tm9             CT1_On          CT1_Off
cutoffs = np.array([cutoff_fastest, cutoff_fastest/10, cutoff_fastest, cutoff_fastest/5, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest])

def test_emd(model, freq, num_cycles, invert=False):
    model.reset()
    size = 7*7
    start = 3*7

    shape = [7, 7]
    stim, y_a = gen_gratings(shape, freq, 'a', num_cycles)

    num_samples = np.shape(stim)[0]

    data = torch.zeros([num_samples, 2*size], device=device)

    for i in range(num_samples):
        data[i,:] = model(stim[i,:])

    data = data.to('cpu')
    data = data.transpose(0,1)
    a = data[:size, :]
    b = data[size:, :]

    a_row = a[start:start+7,:]
    a_single = a_row[3,:]
    b_row = b[start:start+7,:]
    b_single = b_row[3,:]

    if invert:
        a_peak = torch.min(a_single[int(num_samples/2):])
        b_peak = torch.min(b_single[int(num_samples/2):])
    else:
        a_peak = torch.max(a_single[int(num_samples / 2):])
        b_peak = torch.max(b_single[int(num_samples / 2):])
    ratio = a_peak/b_peak

    return a_peak, b_peak, ratio

def freq_response_emd(params, freqs, num_cycles):
    invert = False
    title = 'T4'
    num_freqs = len(freqs)
    a_peaks = np.zeros_like(freqs)
    b_peaks = np.zeros_like(freqs)
    ratios = np.zeros_like(freqs)
    model, _ = gen_emd_on_mcmc(params)
    for i in range(num_freqs):
        print('Sample %i/%i: %f Hz'%(i+1, num_freqs, freqs[i]))
        a_peaks[i], b_peaks[i], ratios[i] = test_emd(model, freqs[i], num_cycles, invert=invert)

    plt.subplot(2,1,1)
    plt.plot(freqs, a_peaks)
    plt.plot(freqs, b_peaks)
    plt.title('A Frequency Response')
    plt.ylabel('Steady Peak')
    plt.xlabel('Frequency (Hz)')
    plt.subplot(2,1,2)
    plt.plot(freqs, ratios)
    plt.title(title+'_a/'+ title+ '_b')
    plt.ylabel('Magnitude')
    plt.xlabel('Frequency (Hz)')

freqs = np.linspace(10,500, num=10)
#                   Retina          L1 low              L1 High         L3              Mi1             Mi9             CT1 On          T4            K_Mi1 K_Mi9 K_CT1 K_T4
params = np.array([cutoff_fastest, cutoff_fastest/10, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, 0.5,  0.5,  0.5,  0.1])

plt.figure()
freq_response_emd(params, freqs, 10)

plt.show()