import numpy as np
import torch
import matplotlib.pyplot as plt
from sns_toolbox.connections import NonSpikingSynapse
from sns_toolbox.networks import Network
from utilities import add_lowpass_filter, add_scaled_bandpass_filter, synapse_target, activity_range, load_data, dt, backend, save_data, reversal_ex

from scipy.optimize import minimize_scalar

def create_net(k, cutoff, params_in, params_bp, params_d_on):
    net = Network()

    add_lowpass_filter(net, params_in['cutoff'], name='In')
    net.add_input('In')

    add_scaled_bandpass_filter(net, params_bp['cutoffLow'], params_bp['cutoffHigh'],
                               params_bp['gain'], invert=params_bp['invert'], name='BP')

    g_in_bp, reversal_in_bp = synapse_target(0.0, activity_range)
    synapse_in_bp = NonSpikingSynapse(max_conductance=g_in_bp, reversal_potential=reversal_in_bp, e_lo=0.0,
                                      e_hi=activity_range)
    net.add_connection(synapse_in_bp, 'In', 'BP_in')

    synapse_bp_d_on = NonSpikingSynapse(max_conductance=params_d_on['g'], reversal_potential=params_d_on['reversal'], e_lo=0.0,
                                        e_hi=activity_range)

    add_lowpass_filter(net, cutoff=params_d_on['cutoff'], name='D On', invert=False, bias=params_d_on['bias'], initial_value=0.0)

    net.add_connection(synapse_bp_d_on, 'BP_out', 'D On')

    reversal_d_on_s_on = reversal_ex
    g_d_on_s_on = k*activity_range / (reversal_d_on_s_on - k*activity_range)

    synapse_d_on_s_on = NonSpikingSynapse(max_conductance=g_d_on_s_on, reversal_potential=reversal_d_on_s_on, e_lo=0.0,
                                          e_hi=activity_range)

    add_lowpass_filter(net, cutoff=cutoff, name='S On', invert=False)

    net.add_connection(synapse_d_on_s_on, 'D On', 'S On')

    net.add_output('S On')

    model = net.compile(dt, backend='numpy', device='cpu')
    return model

def run_net(k, cutoff, params_in, params_bp, params_d_on):
    model = create_net(k, cutoff, params_in, params_bp, params_d_on)
    t = np.arange(0, 50, dt)
    inputs = np.ones([len(t), 1])
    data = np.zeros_like(t)

    for i in range(len(t)):
        data[i] = model(inputs[i, :])
    # plt.plot(t, data, label=str(k))

    return np.max(data)

def error(k, target_peak, cutoff, params_in, params_bp, params_d_on):
    peak = run_net(k, cutoff, params_in, params_bp, params_d_on)
    peak_error = (peak - target_peak) ** 2
    return peak_error

def tune_s_on(cutoff, params_in, params_bp, params_d_on):
    f = lambda x : error(x, 1.0, cutoff, params_in, params_bp, params_d_on)
    res = minimize_scalar(f, bounds=(1.0,2.0), method='bounded')

    k_final = res.x

    params = {'cutoff': cutoff,
              'invert': False,
              'initialValue': 0.0,
              'bias': 0.0,
              'g': k_final * activity_range / (reversal_ex - k_final * activity_range),
              'reversal': reversal_ex}

    return params
