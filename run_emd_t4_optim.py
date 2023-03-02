import numpy as np
import time
from tqdm import tqdm

from utilities import cutoff_fastest, calc_cap_from_cutoff, load_data
from motion_vision_networks import gen_emd_on_mcmc

start = time.time()

data = load_data('mcmc_stims.p')
dt = data['dt']
freqs = data['freqs']
stims = data['stims']

def test_emd(model, stim):
    model.reset()
    size = 7*7
    start = 3*7

    num_samples = np.shape(stim)[0]

    data = np.zeros([num_samples, 2*size])

    for i in (range(num_samples)):
        data[i,:] = model(stim[i,:])

    data = data.transpose()
    a = data[:size, :]
    b = data[size:, :]

    a_row = a[start:start+7,:]
    a_single = a_row[3,:]
    b_row = b[start:start+7,:]
    b_single = b_row[3,:]

    a_peak = np.max(a_single[int(num_samples / 2):])
    b_peak = np.max(b_single[int(num_samples / 2):])
    ratio = a_peak/b_peak

    return a_peak, b_peak, ratio

def freq_response_emd(params):
    cap_fastest = calc_cap_from_cutoff(np.max(params[:8]))
    dt = cap_fastest / 5
    num_freqs = len(freqs)
    a_peaks = np.zeros_like(freqs)
    b_peaks = np.zeros_like(freqs)
    ratios = np.zeros_like(freqs)
    device = 'cpu'
    model, _ = gen_emd_on_mcmc(params, dt, device)
    for i in (range(num_freqs)):
        # print('Sample %i/%i: %f Hz'%(i+1, num_freqs, freqs[i]))
        a_peaks[i], b_peaks[i], ratios[i] = test_emd(model, stims[i])

    return a_peaks, b_peaks, ratios

def cost_function(a_peaks, b_peaks, ratios):
    peak_range = a_peaks[0] - a_peaks[1]
    peak_sums = np.sum(a_peaks)
    ratios_shifted = ratios - 1
    ratios_sum = np.sum(ratios_shifted)
    w0 = 1000
    w1 = w0
    w2 = w0

    cost = -(w0*peak_range + w1*peak_sums + w2*ratios_sum)

    return cost
def evaluate(params):
    a_peaks, b_peaks, ratios = freq_response_emd(params)
    cost = cost_function(a_peaks, b_peaks, ratios)
    return cost

#                   Retina          L1 low              L1 High         L3              Mi1             Mi9             CT1 On          T4            K_Mi1 K_Mi9 K_CT1 K_T4
params = np.array([cutoff_fastest, cutoff_fastest/10, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, cutoff_fastest, 0.5,  0.5,  0.5,  0.1])   # Good guess

cost = evaluate(params)
