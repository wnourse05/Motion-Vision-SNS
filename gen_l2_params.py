import numpy as np
import torch
import matplotlib.pyplot as plt
from sns_toolbox.connections import NonSpikingSynapse
from sns_toolbox.networks import Network
from utilities import add_lowpass_filter, add_scaled_bandpass_filter, synapse_target, activity_range, reversal_in, cutoff_fastest, load_data, dt, backend, save_data

from scipy.optimize import minimize_scalar

def create_net(k, cutoff_low, cutoff_high):
    net = Network()

    params_node_retina = load_data('params_node_retina.p')
    add_lowpass_filter(net, params_node_retina['params']['cutoff'], name='Retina')
    net.add_input('Retina')

    add_scaled_bandpass_filter(net, cutoff_low, cutoff_high, k, invert=True, name='L1')

    g_r_l1, reversal_r_l1 = synapse_target(0.0, activity_range)
    synapse_r_l1 = NonSpikingSynapse(max_conductance=g_r_l1, reversal_potential=reversal_r_l1, e_lo=0.0,
                                     e_hi=activity_range)
    net.add_connection(synapse_r_l1, 'Retina', 'L1_in')

    net.add_output('L1_out')

    model = net.compile(dt, backend=backend, device='cpu')
    return model

def run_net(k, cutoff_low, cutoff_high):
    model = create_net(k, cutoff_low, cutoff_high)
    t = np.arange(0, 50, dt)
    inputs = torch.ones([len(t), 1])
    data = np.zeros_like(t)

    for i in range(len(t)):
        data[i] = model(inputs[i, :])
    plt.plot(t, data, label=str(k))

    return np.min(data)

def error(k, target_peak, cutoff_low, cutoff_high):
    peak = run_net(k, cutoff_low, cutoff_high)
    peak_error = (peak - target_peak) ** 2
    return peak_error

def tune_l2(cutoff_low, cutoff_high):
    f = lambda x : error(x, 0.0, cutoff_low, cutoff_high)
    res = minimize_scalar(f, bounds=(1.0,-reversal_in), method='bounded')

    k_final = res.x
    print('Squared Error: ' + str(res.fun))
    print('Gain: ' + str(k_final))

    type = 'bandpass'
    name = 'L2'
    params = {'cutoffLow': cutoff_low,
              'cutoffHigh': cutoff_high,
              'gain': k_final,
              'invert': True}

    data = {'name': name,
            'type': type,
            'params': params}

    filename = 'params_node_l2.p'

    save_data(data, filename)

cutoff_low = cutoff_fastest/5
cutoff_high = cutoff_fastest

tune_l2(cutoff_low, cutoff_high)