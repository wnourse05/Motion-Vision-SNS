import numpy as np
import torch
import matplotlib.pyplot as plt
from sns_toolbox.connections import NonSpikingSynapse
from sns_toolbox.networks import Network
from utilities import add_lowpass_filter, add_scaled_bandpass_filter, synapse_target, activity_range, load_data, dt, backend, save_data, reversal_ex

from scipy.optimize import minimize_scalar

def create_net(k, cutoff, params_node_retina, params_node_l1, params_node_mi1, params_conn_mi1):
    net = Network()

    add_lowpass_filter(net, params_node_retina['params']['cutoff'], name='Retina')
    net.add_input('Retina')

    add_scaled_bandpass_filter(net, params_node_l1['params']['cutoffLow'], params_node_l1['params']['cutoffHigh'],
                               params_node_l1['params']['gain'], invert=params_node_l1['params']['invert'], name='L1')

    g_r_l1, reversal_r_l1 = synapse_target(0.0, activity_range)
    synapse_r_l1 = NonSpikingSynapse(max_conductance=g_r_l1, reversal_potential=reversal_r_l1, e_lo=0.0,
                                     e_hi=activity_range)
    net.add_connection(synapse_r_l1, 'Retina', 'L1_in')

    add_lowpass_filter(net, cutoff=params_node_mi1['params']['cutoff'], name='Mi1',
                             invert=params_node_mi1['params']['invert'], bias=params_node_mi1['params']['bias'],
                             initial_value=params_node_mi1['params']['initialValue'])
    synapse_l1_mi1 = NonSpikingSynapse(max_conductance=params_conn_mi1['g'],
                                       reversal_potential=params_conn_mi1['reversal'], e_lo=0.0, e_hi=activity_range)
    net.add_connection(synapse_l1_mi1, 'L1_out', 'Mi1')

    reversal_mi1_ct1on = reversal_ex
    g_mi1_ct1on = k*activity_range / (reversal_mi1_ct1on - k*activity_range)

    synapse_mi1_ct1on = NonSpikingSynapse(max_conductance=g_mi1_ct1on, reversal_potential=reversal_mi1_ct1on, e_lo=0.0,
                                          e_hi=activity_range)

    add_lowpass_filter(net, cutoff=cutoff, name='CT1_On', invert=False)

    net.add_connection(synapse_mi1_ct1on, 'Mi1', 'CT1_On')

    net.add_output('CT1_On', name='OutCT1On')

    model = net.compile(dt, backend=backend, device='cpu')
    return model

def run_net(k, cutoff, params_node_retina, params_node_l1, params_node_mi1, params_conn_mi1):
    model = create_net(k, cutoff, params_node_retina, params_node_l1, params_node_mi1, params_conn_mi1)
    t = np.arange(0, 50, dt)
    inputs = torch.ones([len(t), 1])
    data = np.zeros_like(t)

    for i in range(len(t)):
        data[i] = model(inputs[i, :])
    # plt.plot(t, data_sns_toolbox, label=str(k))

    return np.max(data)

def error(k, target_peak, cutoff, params_node_retina, params_node_l1, params_node_mi1, params_conn_mi1):
    peak = run_net(k, cutoff, params_node_retina, params_node_l1, params_node_mi1, params_conn_mi1)
    peak_error = (peak - target_peak) ** 2
    return peak_error

def tune_ct1_on(cutoff, params_node_retina, params_node_l1, params_node_mi1, params_conn_mi1, save=True):
    f = lambda x : error(x, 1.0, cutoff, params_node_retina, params_node_l1, params_node_mi1, params_conn_mi1)
    res = minimize_scalar(f, bounds=(1.0,2.0), method='bounded')

    k_final = res.x
    # print('Squared Error: ' + str(res.fun))
    # print('Gain: ' + str(k_final))

    type = 'lowpass'
    name = 'Mi1'
    params = {'cutoff': cutoff,
              'invert': False,
              'initialValue': 0.0,
              'bias': 0.0}

    data = {'name': name,
            'type': type,
            'params': params}

    filename = '../params_node_ct1_on.p'

    if save:
        save_data(data, filename)

    conn_params = {'source': 'Mi1',
                   'g': k_final*activity_range / (reversal_ex - k_final*activity_range),
                   'reversal': reversal_ex}
    conn_filename = '../params_conn_ct1_on.p'

    if save:
        save_data(conn_params, conn_filename)

    return data, conn_params
