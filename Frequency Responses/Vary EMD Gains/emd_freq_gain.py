import numpy as np
import torch
import pickle

from utilities import cutoff_fastest, device, gen_gratings
from motion_vision_networks import gen_motion_vision

#                   Retina          L1                                  L2                              L3                  Mi1         Mi9             Tm1             Tm9             CT1_On          CT1_Off
cutoffs = np.array([cutoff_fastest, cutoff_fastest/10, cutoff_fastest, cutoff_fastest/5, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest])

def test_emd(model, net, freq, num_cycles, invert=False):
    model.reset()
    size = 7*7
    start = 3*7

    shape = [7, 7]
    stim, y_a = gen_gratings(shape, freq, 'a', num_cycles)

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

def freq_response_emd(gains_left, gains_center, gains_right, freqs, num_cycles, t4=True):
    if t4:
        title = 'T4'
        invert = False
    else:
        title = 'T5'
        invert = True
    num_gains = len(gains_left)
    num_freqs = len(freqs)
    a_peaks = np.zeros([num_gains, num_gains, num_gains, num_freqs])
    ratios = np.zeros_like(a_peaks)
    index = 0
    for l in range(num_gains):
        for c in range(num_gains):
            for r in range(num_gains):
                if t4:
                    model, net = gen_motion_vision((7, 7), k_mi1=gains_center[c], k_mi9=gains_left[l], k_ct1on=gains_right[r],
                                                   output_t4a=True, output_t4b=True)
                else:
                    model, net = gen_motion_vision((7, 7), k_tm1=gains_center[c], k_tm9=gains_left[l], k_ct1off=gains_right[r],
                                                   output_t5a=True, output_t5b=True)
                for f in range(num_freqs):
                    index += 1
                    print('%s Left %i/%i | Center %i/%i | Right %i/%i | Frequency %i/%i | Sample %i/%i' % (title,
                                                                                                           l+1, num_gains,
                                                                                                           c+1, num_gains,
                                                                                                           r+1, num_gains,
                                                                                                           f+1, num_freqs,
                                                                                                           1 + f + r*num_freqs + c*num_freqs*num_gains + l*num_freqs*(num_gains**2),
                                                                                                           (num_gains**3)*num_freqs))
                    a_peaks[l,c,r,f], _, ratios[l,c,r,f] = test_emd(model, net, freqs[f], num_cycles, invert=invert)
    return a_peaks, ratios

freqs = np.linspace(10,500, num=10)
gains = np.linspace(0.01,0.5,num=5)
print('Run 1/2: T4')
t4_peaks, t4_ratios = freq_response_emd(gains, gains, gains, freqs, 10, t4=True)
print('Run 2/2: T5')
t5_peaks, t5_ratios = freq_response_emd(gains, gains, gains, freqs, 10, t4=False)

data = {'peaksT4': t4_peaks,
        'peaksT5': t5_peaks,
        'ratiosT4': t4_ratios,
        'ratiosT5': t5_ratios,
        'gains': gains,
        'frequencies': freqs}

pickle.dump(data, open('emd_freq_gain_data.p', 'wb'))
