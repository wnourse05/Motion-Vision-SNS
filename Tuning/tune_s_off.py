import numpy as np
import torch
import matplotlib.pyplot as plt
from sns_toolbox.connections import NonSpikingSynapse
from sns_toolbox.networks import Network
from utilities import add_lowpass_filter, add_scaled_bandpass_filter, synapse_target, activity_range, reversal_in, cutoff_fastest, load_data, dt, backend, save_data, reversal_ex

from scipy.optimize import minimize_scalar

def create_net(g_d_off_s_off, cutoff, params_in, params_bp, params_d_off):
    net = Network()

    add_lowpass_filter(net, params_in['cutoff'], name='In')
    net.add_input('In')

    add_scaled_bandpass_filter(net, params_bp['cutoffLow'], params_bp['cutoffHigh'],
                               params_bp['gain'], invert=params_bp['invert'], name='BP')

    g_in_bp, reversal_in_bp = synapse_target(0.0, activity_range)
    synapse_in_bp = NonSpikingSynapse(max_conductance=g_in_bp, reversal_potential=reversal_in_bp, e_lo=0.0,
                                     e_hi=activity_range)
    net.add_connection(synapse_in_bp, 'In', 'BP_in')

    synapse_bp_d_off = NonSpikingSynapse(max_conductance=params_d_off['g'], reversal_potential=params_d_off['reversal'], e_lo=activity_range,
                                       e_hi=2*activity_range)

    add_lowpass_filter(net, cutoff=cutoff, name='D Off', invert=False, bias=0.0, initial_value=0.0)

    net.add_connection(synapse_bp_d_off, 'BP_out', 'D Off')

    # g_d_off_s_off, reversal_d_off_s_off = synapse_target(activity_range, bias)
    reversal_d_off_s_off = reversal_ex

    synapse_tm1_ct1off = NonSpikingSynapse(max_conductance=g_d_off_s_off, reversal_potential=reversal_d_off_s_off, e_lo=0.0, e_hi=activity_range)

    add_lowpass_filter(net, cutoff=cutoff, name='S Off', invert=False, initial_value=0.0, bias=0.0)

    net.add_connection(synapse_tm1_ct1off, 'D Off', 'S Off')

    net.add_output('S Off')

    model = net.compile(dt, backend='numpy', device='cpu')
    return model

def run_net(g, cutoff, params_in, params_bp, params_d_off):
    model = create_net(g, cutoff, params_in, params_bp, params_d_off)
    t = np.arange(0, 50, dt)
    inputs = np.ones([len(t), 1])
    inputs[int(len(t)/2):, :] = 0.0
    data = np.zeros_like(t)

    for i in range(len(t)):
        data[i] = model(inputs[i, :])
    # plt.plot(t, data, label=str(bias))

    return np.max(data)

def error(g, target_peak, cutoff, params_in, params_bp, params_d_off):
    peak = run_net(g, cutoff, params_in, params_bp, params_d_off)
    peak_error = (peak - target_peak) ** 2
    return peak_error

def tune_s_off(cutoff, params_in, params_bp, params_d_off):
    f = lambda x : error(x, 1.0, cutoff, params_in, params_bp, params_d_off)
    res = minimize_scalar(f, bounds=(0.0, 10.0), method='bounded')

    g_final = res.x

    params = {'cutoff': cutoff,
              'invert': False,
              'initialValue': 0.0,
              'bias': 0.0,
               'g': g_final,
               'reversal': reversal_ex}

    return params
