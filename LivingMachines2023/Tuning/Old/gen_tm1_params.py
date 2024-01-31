import numpy as np
import torch
import matplotlib.pyplot as plt
from sns_toolbox.connections import NonSpikingSynapse
from sns_toolbox.networks import Network
from utilities import add_lowpass_filter, add_scaled_bandpass_filter, synapse_target, activity_range, load_data, dt, backend, save_data, reversal_ex

from scipy.optimize import minimize_scalar

def create_net(g_l2_tm1, cutoff, params_node_retina, params_node_l2):
    net = Network()
    bias = 0.0

    add_lowpass_filter(net, params_node_retina['params']['cutoff'], name='Retina')
    net.add_input('Retina')

    add_scaled_bandpass_filter(net, params_node_l2['params']['cutoffLow'], params_node_l2['params']['cutoffHigh'],
                               params_node_l2['params']['gain'], invert=params_node_l2['params']['invert'], name='L2')

    g_r_l2, reversal_r_l2 = synapse_target(0.0, activity_range)
    synapse_r_l2 = NonSpikingSynapse(max_conductance=g_r_l2, reversal_potential=reversal_r_l2, e_lo=0.0,
                                     e_hi=activity_range)
    net.add_connection(synapse_r_l2, 'Retina', 'L2_in')

    # g_bp_d_off, reversal_l2_tm1 = synapse_target(activity_range, bias)
    reversal_l2_tm1 = reversal_ex

    synapse_l2_tm1 = NonSpikingSynapse(max_conductance=g_l2_tm1, reversal_potential=reversal_l2_tm1, e_lo=activity_range,
                                       e_hi=2*activity_range)

    add_lowpass_filter(net, cutoff=cutoff, name='Tm1', invert=False, bias=bias, initial_value=0.0)

    net.add_connection(synapse_l2_tm1, 'L2_out', 'Tm1')

    net.add_output('Tm1')

    model = net.compile(dt, backend=backend, device='cpu')
    return model

def run_net(g, cutoff, params_node_retina, params_node_l2):
    model = create_net(g, cutoff, params_node_retina, params_node_l2)
    t = np.arange(0, 50, dt)
    inputs = torch.ones([len(t), 1])
    inputs[int(len(t)/2):, :] = 0.0
    data = np.zeros_like(t)

    for i in range(len(t)):
        data[i] = model(inputs[i, :])
    # plt.plot(t, data_sns_toolbox, label=str(bias))

    return np.max(data)

def error(g, target_peak, cutoff, params_node_retina, params_node_l2):
    peak = run_net(g, cutoff, params_node_retina, params_node_l2)
    peak_error = (peak - target_peak) ** 2
    return peak_error

def tune_tm1(cutoff, params_node_retina, params_node_l2, save=True):
    f = lambda x : error(x, 1.0, cutoff, params_node_retina, params_node_l2)
    res = minimize_scalar(f, bounds=(0.0, 10.0), method='bounded')

    g_final = res.x
    # print(g_final)
    # print('Squared Error: ' + str(res.fun))
    # print('Bias: ' + str(bias_final) + ' nA')

    type = 'lowpass'
    name = 'Tm1'
    params = {'cutoff': cutoff,
              'invert': False,
              'initialValue': 0.0,
              'bias': 0}

    data = {'name': name,
            'type': type,
            'params': params}

    filename = '../params_node_tm1.p'
    if save:
        save_data(data, filename)
    # g_bp_d_off, reversal_l2_tm1 = synapse_target(activity_range, bias_final)
    conn_params = {'source': 'L2',
                   'g': g_final,#g_bp_d_off,
                   'reversal': reversal_ex}#reversal_l2_tm1}
    conn_filename = '../params_conn_tm1.p'
    if save:
        save_data(conn_params, conn_filename)

    return data, conn_params
