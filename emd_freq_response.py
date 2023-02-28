import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib

from utilities import dt, cutoff_fastest, Stimulus, device, gen_gratings
from motion_vision_networks import gen_test_emd
from sns_toolbox.renderer import render

#                   Retina          L1                                  L2                              L3                  Mi1         Mi9             Tm1             Tm9             CT1_On          CT1_Off
cutoffs = np.array([cutoff_fastest, cutoff_fastest/10, cutoff_fastest, cutoff_fastest/5, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest])

def test_emd(model, net, freq, num_cycles, invert=False):
    model.reset()
    size = 7*7
    start = 3*7

    shape = [7, 7]
    stim, y_a = gen_gratings(shape, freq, 'a', num_cycles)

    # stim = Stimulus(stimulus, interval)
    num_samples = np.shape(stim)[0]

    data = torch.zeros([num_samples, net.get_num_outputs_actual()], device=device)

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

def freq_response_emd(model, net, freqs, num_cycles, title, invert=False):
    num_freqs = len(freqs)
    a_peaks = np.zeros_like(freqs)
    b_peaks = np.zeros_like(freqs)
    ratios = np.zeros_like(freqs)

    for i in range(num_freqs):
        print('Sample %i/%i: %f Hz'%(i+1, num_freqs, freqs[i]))
        a_peaks[i], b_peaks[i], ratios[i] = test_emd(model, net, freqs[i], num_cycles, invert=invert)

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(freqs, a_peaks, label=title+'_a')
    plt.plot(freqs, b_peaks, label=title + '_b')
    plt.title('Frequency Response')
    plt.legend()
    plt.ylabel('Steady Peak')
    plt.xlabel('Frequency (Hz)')
    plt.subplot(2,1,2)
    plt.plot(freqs, ratios)
    plt.title(title+'_a/'+ title+ '_b')
    plt.ylabel('Magnitude')
    plt.xlabel('Frequency (Hz)')

model_t4, net_t4 = gen_test_emd((7,7), output_t4a=True, output_t4b=True)
model_t5, net_t5 = gen_test_emd((7,7), output_t5a=True, output_t5b=True)

freqs = np.linspace(1,1000, num=50)
# freq_response_emd(model_t4, net_t4, freqs, 10, 'T4')
freq_response_emd(model_t5, net_t5, freqs, 10, 'T5', invert=True)

plt.show()