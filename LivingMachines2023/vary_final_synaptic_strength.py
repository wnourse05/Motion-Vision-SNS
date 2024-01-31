import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from sns_toolbox.neurons import NonSpikingNeuron
from sns_toolbox.connections import NonSpikingSynapse
from sns_toolbox.networks import Network
from utilities import add_lowpass_filter, bandpass_filter, synapse_target, calc_cap_from_cutoff, activity_range, reversal_ex, reversal_in

from scipy.optimize import minimize_scalar

def add_scaled_bandpass_filter(net: Network, cutoff_lower, cutoff_higher, k, invert=False, name=None):
    if name is None:
        name = 'bandpass'
    if invert:
        rest = activity_range
        g_in = (-activity_range)/reversal_in
        g_bd = (-k*activity_range)/(reversal_in + k*activity_range)
        g_cd = (g_bd*(reversal_in-activity_range))/(activity_range-reversal_ex)
        synapse_fast = NonSpikingSynapse(max_conductance=g_in, reversal_potential=reversal_in, e_lo=0.0, e_hi=activity_range)
        synapse_bd = NonSpikingSynapse(max_conductance=g_bd, reversal_potential=reversal_in, e_lo=0.0, e_hi=activity_range)
        synapse_slow = NonSpikingSynapse(max_conductance=g_cd, reversal_potential=reversal_ex, e_lo=0.0, e_hi=activity_range)
    else:
        rest = 0.0
        g_ex = activity_range/(reversal_ex-activity_range)
        g_bd = k*activity_range/(reversal_ex-k*activity_range)
        g_cd = (-g_bd*reversal_ex)/reversal_in
        synapse_fast = NonSpikingSynapse(max_conductance=g_ex, reversal_potential=reversal_ex, e_lo=0.0, e_hi=activity_range)
        synapse_bd = NonSpikingSynapse(max_conductance=g_bd, reversal_potential=reversal_ex, e_lo=0.0, e_hi=activity_range)
        synapse_slow = NonSpikingSynapse(max_conductance=g_cd, reversal_potential=reversal_in, e_lo=0.0, e_hi=activity_range)

    neuron_fast = NonSpikingNeuron(membrane_conductance=1.0, membrane_capacitance=calc_cap_from_cutoff(cutoff_higher), resting_potential=rest)
    neuron_slow = NonSpikingNeuron(membrane_conductance=1.0, membrane_capacitance=calc_cap_from_cutoff(cutoff_lower), resting_potential=rest)

    net.add_neuron(neuron_fast, name=name+'_in')
    net.add_neuron(neuron_fast, name=name+'_fast', initial_value=0.0)
    net.add_neuron(neuron_slow, name=name+'_slow', initial_value=0.0)
    net.add_neuron(neuron_fast, name=name+'_out')

    net.add_connection(synapse_fast, name+'_in', name+'_fast')
    net.add_connection(synapse_fast, name+'_in', name+'_slow')
    net.add_connection(synapse_bd, name+'_fast', name+'_out')
    net.add_connection(synapse_slow, name+'_slow', name+'_out')

def create_network(k):
    net = Network()
    cutoff = 200
    c_fastest = calc_cap_from_cutoff(cutoff)
    dt = c_fastest / 10

    retina = add_lowpass_filter(net, cutoff, name='Retina')

    net.add_input('Retina')

    add_scaled_bandpass_filter(net, cutoff / 10, cutoff, k, invert=True)

    g_r_l1, reversal_r_l1 = synapse_target(0.0, activity_range)
    synapse_r_l1 = NonSpikingSynapse(max_conductance=g_r_l1, reversal_potential=reversal_r_l1, e_lo=0.0,
                                     e_hi=activity_range)
    net.add_connection(synapse_r_l1, 'Retina', 'bandpass_in')

    net.add_output('bandpass_out')

    model = net.compile(dt, backend='torch', device='cpu')
    return model, dt

def test_net(k):
    model, dt = create_network(k)

    t = np.arange(0, 50, dt)
    inputs = torch.ones([len(t), 1])
    data = np.zeros_like(t)

    for i in range(len(t)):
        data[i] = model(inputs[i,:])
    plt.plot(t, data, label=str(k))

    return np.min(data)

def error(k, target_peak):
    peak = test_net(k)
    peak_error = (peak - target_peak)**2
    return peak_error


k = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8]#, 1.9]
plt.figure()
for val in k:
    test_net(val)

plt.legend()

plt.figure()
f = lambda x : error(x, 0.0)
res = minimize_scalar(f, bounds=(1.0,2.0), method='bounded')
plt.legend()

peak_error = res.fun
k_final = res.x
plt.figure()
peak = test_net(k_final)

print(peak_error)
print(k_final)
plt.show()
