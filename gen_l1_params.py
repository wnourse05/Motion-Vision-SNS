import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from sns_toolbox.neurons import NonSpikingNeuron
from sns_toolbox.connections import NonSpikingSynapse
from sns_toolbox.networks import Network
from utilities import add_lowpass_filter, add_scaled_bandpass_filter, synapse_target, calc_cap_from_cutoff, activity_range, reversal_ex, reversal_in, cutoff_fastest, load_data, dt, backend, save_data

from scipy.optimize import minimize_scalar

cutoff_low = cutoff_fastest/10
cutoff_high = cutoff_fastest

def create_net(k):
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

def run_net(k):
    model = create_net(k)
    t = np.arange(0, 50, dt)
    inputs = torch.ones([len(t), 1])
    data = np.zeros_like(t)

    for i in range(len(t)):
        data[i] = model(inputs[i, :])
    plt.plot(t, data, label=str(k))

    return np.min(data)

def error(k, target_peak):
    peak = run_net(k)
    peak_error = (peak - target_peak) ** 2
    return peak_error

f = lambda x : error(x, 0.0)
res = minimize_scalar(f, bounds=(1.0,-reversal_in), method='bounded')

k_final = res.x
print('Squared Error: ' + str(res.fun))
print('Gain: ' + str(k_final))

type = 'bandpass'
name = 'L1'
params = {'cutoffLow': cutoff_low,
          'cutoffHigh': cutoff_high,
          'gain': k_final,
          'invert': True}

data = {'name': name,
        'type': type,
        'params': params}

filename = 'params_node_l1.p'

save_data(data, filename)
