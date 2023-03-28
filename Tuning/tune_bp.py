import numpy as np
import torch
import matplotlib.pyplot as plt
from sns_toolbox.connections import NonSpikingSynapse
from sns_toolbox.networks import Network
from utilities import add_lowpass_filter, add_scaled_bandpass_filter, synapse_target, activity_range, reversal_in, load_data, dt, backend, save_data

from scipy.optimize import minimize_scalar

def create_net(k, cutoff_fast, cutoff_low):
    net = Network()

    add_lowpass_filter(net, cutoff_fast, name='In')
    net.add_input('In')

    add_scaled_bandpass_filter(net, cutoff_low, cutoff_fast, k, invert=True, name='BP')

    g_in_bp, reversal_in_bp = synapse_target(0.0, activity_range)
    synapse_in_bp = NonSpikingSynapse(max_conductance=g_in_bp, reversal_potential=reversal_in_bp, e_lo=0.0,
                                     e_hi=activity_range)
    net.add_connection(synapse_in_bp, 'In', 'BP_in')

    net.add_output('BP_out')

    model = net.compile(dt, backend='numpy', device='cpu')
    return model

def run_net(k, cutoff_fast, cutoff_low):
    model = create_net(k, cutoff_fast, cutoff_low)
    t = np.arange(0, 50, dt)
    inputs = np.ones([len(t), 1])
    data = np.zeros_like(t)

    for i in range(len(t)):
        data[i] = model(inputs[i, :])
    # plt.plot(t, data, label=str(k))

    return np.min(data)

def error(k, target_peak, cutoff_fast, cutoff_low):
    peak = run_net(k, cutoff_fast, cutoff_low)
    peak_error = (peak - target_peak) ** 2
    return peak_error

def tune_bp(cutoff_fast, cutoff_low):
    f = lambda x : error(x, 0.0, cutoff_fast, cutoff_low)
    res = minimize_scalar(f, bounds=(1.0,-reversal_in), method='bounded')

    k_final = res.x

    target_center = 0.0
    target_middle = 7 / 8 * activity_range
    target_outer = 1.0 * activity_range
    outer_conductance, outer_reversal = synapse_target(target_outer, activity_range)
    middle_conductance, middle_reversal = synapse_target(target_middle, activity_range)
    center_conductance, center_reversal = synapse_target(target_center, activity_range)
    g = {'outer': outer_conductance,
         'middle': middle_conductance,
         'center': center_conductance}
    reversal = {'outer': outer_reversal,
                'middle': middle_reversal,
                'center': center_reversal}

    params = {'cutoffLow': cutoff_low,
              'cutoffHigh': cutoff_fast,
              'gain': k_final,
              'invert': True,
              'g': g,
              'reversal': reversal}

    return params