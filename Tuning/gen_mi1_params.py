import numpy as np
import torch
import matplotlib.pyplot as plt
from sns_toolbox.connections import NonSpikingSynapse
from sns_toolbox.networks import Network
from utilities import add_lowpass_filter, add_scaled_bandpass_filter, synapse_target, activity_range, reversal_in, load_data, dt, backend, save_data

from scipy.optimize import minimize_scalar

def create_net(bias, cutoff):
    net = Network()

    params_node_retina = load_data('../params_node_retina.p')
    add_lowpass_filter(net, params_node_retina['params']['cutoff'], name='Retina')
    net.add_input('Retina')

    params_node_l1 = load_data('../params_node_l1.p')
    add_scaled_bandpass_filter(net, params_node_l1['params']['cutoffLow'], params_node_l1['params']['cutoffHigh'],
                               params_node_l1['params']['gain'], invert=params_node_l1['params']['invert'], name='L1')

    g_r_l1, reversal_r_l1 = synapse_target(0.0, activity_range)
    synapse_r_l1 = NonSpikingSynapse(max_conductance=g_r_l1, reversal_potential=reversal_r_l1, e_lo=0.0,
                                     e_hi=activity_range)
    net.add_connection(synapse_r_l1, 'Retina', 'L1_in')

    g_l1_mi1 = -bias/reversal_in
    reversal_l1_mi1 = reversal_in

    synapse_l1_mi1 = NonSpikingSynapse(max_conductance=g_l1_mi1, reversal_potential=reversal_l1_mi1, e_lo=0.0,
                                       e_hi=activity_range)

    add_lowpass_filter(net, cutoff=cutoff, name='Mi1', invert=False, bias=bias, initial_value=0.0)

    net.add_connection(synapse_l1_mi1, 'L1_out', 'Mi1')

    net.add_output('Mi1')

    model = net.compile(dt, backend=backend, device='cpu')
    return model

def run_net(bias, cutoff):
    model = create_net(bias, cutoff)
    t = np.arange(0, 50, dt)
    inputs = torch.ones([len(t), 1])
    data = np.zeros_like(t)

    for i in range(len(t)):
        data[i] = model(inputs[i, :])
    # plt.plot(t, data, label=str(bias))

    return np.max(data)

def error(bias, target_peak, cutoff):
    peak = run_net(bias, cutoff)
    peak_error = (peak - target_peak) ** 2
    return peak_error

def tune_mi1(cutoff, save=True):
    f = lambda x : error(x, 1.0, cutoff)
    res = minimize_scalar(f, bounds=(0.0,2.0), method='bounded')

    bias_final = res.x
    # print('Squared Error: ' + str(res.fun))
    # print('Bias: ' + str(bias_final) + ' nA')

    type = 'lowpass'
    name = 'Mi1'
    params = {'cutoff': cutoff,
              'invert': False,
              'initialValue': 0.0,
              'bias': bias_final}

    data = {'name': name,
            'type': type,
            'params': params}

    filename = '../params_node_mi1.p'
    if save:
        save_data(data, filename)

    conn_params = {'source': 'L1',
                   'g': -bias_final/reversal_in,
                   'reversal': reversal_in}
    conn_filename = '../params_conn_mi1.p'
    if save:
        save_data(conn_params, conn_filename)

    return data, conn_params
