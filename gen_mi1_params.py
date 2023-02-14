import numpy as np
import torch
import matplotlib.pyplot as plt
from sns_toolbox.connections import NonSpikingSynapse
from sns_toolbox.networks import Network
from utilities import add_lowpass_filter, add_scaled_bandpass_filter, synapse_target, activity_range, reversal_in, cutoff_fastest, load_data, dt, backend, save_data

from scipy.optimize import minimize_scalar

def create_net(g, cutoff):
    net = Network()

    params_node_retina = load_data('params_node_retina.p')
    add_lowpass_filter(net, params_node_retina['params']['cutoff'], name='Retina')
    net.add_input('Retina')

    params_node_l1 = load_data('params_node_l1.p')
    add_scaled_bandpass_filter(net, params_node_l1['params']['cutoffLow'], params_node_l1['params']['cutoffHigh'],
                               params_node_l1['params']['gain'], invert=params_node_l1['params']['invert'], name='L1')

    g_r_l1, reversal_r_l1 = synapse_target(0.0, activity_range)
    synapse_r_l1 = NonSpikingSynapse(max_conductance=g_r_l1, reversal_potential=reversal_r_l1, e_lo=0.0,
                                     e_hi=activity_range)
    net.add_connection(synapse_r_l1, 'Retina', 'L1_in')

    g_l1_mi1 = g
    reversal_l1_mi1 = reversal_in

    synapse_l1_mi1 = NonSpikingSynapse(max_conductance=g_l1_mi1, reversal_potential=reversal_l1_mi1, e_lo=0.0,
                                       e_hi=activity_range)

    add_lowpass_filter(net, cutoff=cutoff, name='Mi1', invert=False, bias=activity_range)

    net.add_connection(synapse_l1_mi1, 'L1_out', 'Mi1')

    net.add_output('Mi1')

    model = net.compile(dt, backend=backend, device='cpu')
    return model

def run_net(g, cutoff):
    model = create_net(g, cutoff)
    t = np.arange(0, 50, dt)
    inputs = torch.ones([len(t), 1])
    data = np.zeros_like(t)

    for i in range(len(t)):
        data[i] = model(inputs[i, :])
    plt.plot(t, data, label=str(g))

    return np.max(data)

def error(g, target_peak, cutoff):
    peak = run_net(g, cutoff)
    peak_error = (peak - target_peak) ** 2
    return peak_error

def tune_mi1(cutoff):
    f = lambda x : error(x, 1.0, cutoff)
    res = minimize_scalar(f, bounds=(0.5,2.0), method='bounded')

    g_final = res.x
    print('Squared Error: ' + str(res.fun))
    print('Conductance: ' + str(g_final) + ' uS')

    type = 'lowpass'
    name = 'Mi1'
    params = {'cutoff': cutoff,
              'invert': True,
              'initialValue': 0.0,
              'bias': activity_range}

    data = {'name': name,
            'type': type,
            'params': params}

    filename = 'params_node_mi1.p'

    save_data(data, filename)

    conn_params = {'source': 'L1',
                   'g': g_final,
                   'reversal': reversal_in}
    conn_filename = 'params_conn_mi1.p'

    save_data(conn_params, conn_filename)


cutoff = cutoff_fastest

tune_mi1(cutoff)
