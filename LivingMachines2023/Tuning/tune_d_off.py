import numpy as np
import torch
import matplotlib.pyplot as plt
from sns_toolbox.connections import NonSpikingSynapse
from sns_toolbox.networks import Network
from utilities import add_lowpass_filter, add_scaled_bandpass_filter, synapse_target, activity_range, load_data, dt, backend, save_data, reversal_ex

from scipy.optimize import minimize_scalar

def create_net(g_bp_d_off, cutoff, params_in, params_bp):
    net = Network()
    bias = 0.0

    add_lowpass_filter(net, params_in['cutoff'], name='In')
    net.add_input('In')

    add_scaled_bandpass_filter(net, params_bp['cutoffLow'], params_bp['cutoffHigh'],
                               params_bp['gain'], invert=params_bp['invert'], name='BP')

    g_in_bp, reversal_in_bp = synapse_target(0.0, activity_range)
    synapse_in_bp = NonSpikingSynapse(max_conductance=g_in_bp, reversal_potential=reversal_in_bp, e_lo=0.0,
                                     e_hi=activity_range)
    net.add_connection(synapse_in_bp, 'In', 'BP_in')

    # g_bp_d_off, reversal_bp_d_off = synapse_target(activity_range, bias)
    reversal_bp_d_off = reversal_ex

    synapse_bp_d_off = NonSpikingSynapse(max_conductance=g_bp_d_off, reversal_potential=reversal_bp_d_off, e_lo=activity_range,
                                       e_hi=2*activity_range)

    add_lowpass_filter(net, cutoff=cutoff, name='D Off', invert=False, bias=bias, initial_value=0.0)

    net.add_connection(synapse_bp_d_off, 'BP_out', 'D Off')

    net.add_output('D Off')

    model = net.compile(dt, backend='numpy', device='cpu')
    return model

def run_net(g, cutoff, params_in, params_bp):
    model = create_net(g, cutoff, params_in, params_bp)
    t = np.arange(0, 50, dt)
    inputs = np.ones([len(t), 1])
    inputs[int(len(t)/2):, :] = 0.0
    data = np.zeros_like(t)

    for i in range(len(t)):
        data[i] = model(inputs[i, :])
    # plt.plot(t, data_sns_toolbox, label=str(bias))

    return np.max(data)

def error(g, target_peak, cutoff, params_in, params_bp):
    peak = run_net(g, cutoff, params_in, params_bp)
    peak_error = (peak - target_peak) ** 2
    return peak_error

def tune_d_off(cutoff, params_in, params_bp):
    f = lambda x : error(x, 1.0, cutoff, params_in, params_bp)
    res = minimize_scalar(f, bounds=(0.0, 10.0), method='bounded')

    g_final = res.x
    # print(g_final)
    # print('Squared Error: ' + str(res.fun))
    # print('Bias: ' + str(bias_final) + ' nA')

    params = {'cutoff': cutoff,
              'invert': False,
              'initialValue': 0.0,
              'bias': 0,
              'g': g_final,
              'reversal': reversal_ex}

    return params
