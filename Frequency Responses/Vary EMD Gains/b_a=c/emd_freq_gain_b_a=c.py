import numpy as np
import torch
import matplotlib.pyplot as plt

from utilities import cutoff_fastest, device, gen_gratings
from motion_vision_networks import gen_test_emd

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

def freq_response_emd(gain_left, gains_center, gain_right, freqs, num_cycles, t4=True):
    if t4:
        invert = False
        title = 'T4'
    else:
        invert = True
        title = 'T5'
    num_gains = len(gains_center)
    num_freqs = len(freqs)
    a_peaks = np.zeros_like(freqs)
    b_peaks = np.zeros_like(freqs)
    ratios = np.zeros_like(freqs)
    for g in range(num_gains):
        print('     Gain %i/%i'%(g+1, num_gains))
        if t4:
            model, net = gen_test_emd((7,7), k_mi1=gains_center[g], k_mi9=gain_left, k_ct1on=gain_right,
                                      output_t4a=True, output_t4b=True)
        else:
            model, net = gen_test_emd((7, 7), k_tm1=gains_center[g], k_tm9=gain_left, k_ct1off=gain_right,
                                      output_t5a=True, output_t5b=True)
        for i in range(num_freqs):
            print('          Sample %i/%i: %f Hz'%(i+1, num_freqs, freqs[i]))
            a_peaks[i], b_peaks[i], ratios[i] = test_emd(model, net, freqs[i], num_cycles, invert=invert)

        plt.subplot(2,1,1)
        plt.plot(freqs, a_peaks, label=str(gains_center[g]), color='C' + str(g))
        plt.plot(freqs, b_peaks, linestyle='--', color='C'+str(g))
        plt.title('A Frequency Response')
        plt.ylabel('Steady Peak')
        plt.xlabel('Frequency (Hz)')
        plt.subplot(2,1,2)
        plt.plot(freqs, ratios, label=str(gains_center[g]))
        plt.title(title+'_a/'+ title+ '_b')
        plt.ylabel('Magnitude')
        plt.xlabel('Frequency (Hz)')
    plt.subplot(2,1,1)
    plt.legend()
    plt.subplot(2,1,2)
    plt.legend()
    plt.suptitle(title +': a=c=' + str(gain_left))

model_t4, net_t4 = gen_test_emd((7,7), output_t4a=True, output_t4b=True)
model_t5, net_t5 = gen_test_emd((7,7), output_t5a=True, output_t5b=True)

freqs = np.linspace(10,500, num=10)
gains = np.linspace(0.01,0.5,num=5)
for i in range(len(gains)):
    print('Run %i/%i: T4'%(i+1, len(gains)))
    plt.figure()
    freq_response_emd(gains[i], gains, gains[i], freqs, 10, t4=True)
    print('Run %i/%i: T5' % (i + 1, len(gains)))
    plt.figure()
    freq_response_emd(gains[i], gains, gains[i], freqs, 10, t4=False)

plt.show()