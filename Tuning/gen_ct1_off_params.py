import numpy as np
import torch
import matplotlib.pyplot as plt
from sns_toolbox.connections import NonSpikingSynapse
from sns_toolbox.networks import Network
from utilities import add_lowpass_filter, add_scaled_bandpass_filter, synapse_target, activity_range, reversal_in, cutoff_fastest, load_data, dt, backend, save_data

from scipy.optimize import minimize_scalar

def create_net(bias, cutoff):
    net = Network()

    params_node_retina = load_data('../params_node_retina.p')
    add_lowpass_filter(net, params_node_retina['params']['cutoff'], name='Retina')
    net.add_input('Retina')

    params_node_l2 = load_data('../params_node_l2.p')
    add_scaled_bandpass_filter(net, params_node_l2['params']['cutoffLow'], params_node_l2['params']['cutoffHigh'],
                               params_node_l2['params']['gain'], invert=params_node_l2['params']['invert'], name='L2')

    g_r_l2, reversal_r_l2 = synapse_target(0.0, activity_range)
    synapse_r_l2 = NonSpikingSynapse(max_conductance=g_r_l2, reversal_potential=reversal_r_l2, e_lo=0.0,
                                     e_hi=activity_range)
    net.add_connection(synapse_r_l2, 'Retina', 'L2_in')

    params_conn_tm1 = load_data('../params_conn_tm1.p')
    synapse_l2_tm1 = NonSpikingSynapse(max_conductance=params_conn_tm1['g'],
                                       reversal_potential=params_conn_tm1['reversal'], e_lo=0.0, e_hi=activity_range)
    params_node_tm1 = load_data('../params_node_tm1.p')
    add_lowpass_filter(net, cutoff=params_node_tm1['params']['cutoff'], name='Tm1',
                             invert=params_node_tm1['params']['invert'], bias=params_node_tm1['params']['bias'],
                             initial_value=params_node_tm1['params']['initialValue'])
    net.add_connection(synapse_l2_tm1, 'L2_out', 'Tm1')

    g_tm1_ct1_off, reversal_tm1_ct1_off = synapse_target(activity_range, bias)

    synapse_tm1_ct1off = NonSpikingSynapse(max_conductance=g_tm1_ct1_off, reversal_potential=reversal_tm1_ct1_off, e_lo=0.0, e_hi=activity_range)

    add_lowpass_filter(net, cutoff=cutoff, name='CT1_Off', invert=False, initial_value=activity_range, bias=bias)

    net.add_connection(synapse_tm1_ct1off, 'Tm1', 'CT1_Off')

    net.add_output('CT1_Off')

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

    return np.min(data)

def error(bias, target_peak, cutoff):
    peak = run_net(bias, cutoff)
    peak_error = (peak - target_peak) ** 2
    return peak_error

def tune_ct1_off(cutoff, save=True):
    f = lambda x : error(x, 0.0, cutoff)
    res = minimize_scalar(f, bounds=(-1.0, 0.0), method='bounded')

    bias_final = res.x
    print('Squared Error: ' + str(res.fun))
    print('Bias: ' + str(bias_final) + ' nA')

    type = 'lowpass'
    name = 'CT1_Off'
    params = {'cutoff': cutoff,
              'invert': False,
              'initialValue': activity_range,
              'bias': bias_final}

    data = {'name': name,
            'type': type,
            'params': params}

    filename = '../params_node_ct1_off.p'

    if save:
        save_data(data, filename)
    g, reversal = synapse_target(activity_range, bias_final)
    conn_params = {'source': 'L2',
                   'g': g,
                   'reversal': reversal}
    conn_filename = '../params_conn_ct1_off.p'

    if save:
        save_data(conn_params, conn_filename)

    return data, conn_params
