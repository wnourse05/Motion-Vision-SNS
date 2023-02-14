import pickle
import numpy as np

from sns_toolbox.neurons import NonSpikingNeuron
from sns_toolbox.connections import NonSpikingSynapse
from sns_toolbox.networks import Network

c_fast = 1.0
activity_range = 1.0
reversal_ex = 5.0
reversal_in = -2.0
backend = 'torch'
cutoff_fastest = 200   # Hz

def save_data(data, filename):
    pickle.dump(data, open(filename, 'wb'))

def load_data(filename):
    data = pickle.load(open(filename, 'rb'))
    return data

def synapse_target(target, bias):
    if target > bias:
        reversal = reversal_ex
    else:
        reversal = reversal_in
    conductance = (bias - target)/(target - reversal)
    return conductance, reversal

def calc_cap_from_cutoff(cutoff):
    cap = 1000/(2*np.pi*cutoff)
    return cap
def add_lowpass_filter(net: Network, cutoff, invert=False, name=None, bias=0.0, initial_value=0.0):
    membrane_conductance = 1.0
    membrane_capacitance = calc_cap_from_cutoff(cutoff)
    if invert:
        rest = activity_range
    else:
        rest = 0.0
    if name is None:
        name = 'lowpass'
    neuron_type = NonSpikingNeuron(membrane_conductance=membrane_conductance, membrane_capacitance=membrane_capacitance, resting_potential=rest, bias=bias)
    net.add_neuron(neuron_type, name=name, initial_value=initial_value)

def bandpass_filter(net: Network, cutoff_lower, cutoff_higher, invert=False, name=None):
    if name is None:
        name = 'bandpass'
    if invert:
        rest = activity_range
        g_in = (-activity_range)/reversal_in
        g_ex = (g_in*(reversal_in-activity_range))/(activity_range-reversal_ex)
        synapse_fast = NonSpikingSynapse(max_conductance=g_in, reversal_potential=reversal_in, e_lo=0.0, e_hi=activity_range)
        synapse_slow = NonSpikingSynapse(max_conductance=g_ex, reversal_potential=reversal_ex, e_lo=0.0, e_hi=activity_range)
    else:
        rest = 0.0
        g_ex = activity_range/(reversal_ex-activity_range)
        g_in = (-g_ex*reversal_ex)/reversal_in
        synapse_fast = NonSpikingSynapse(max_conductance=g_ex, reversal_potential=reversal_ex, e_lo=0.0, e_hi=activity_range)
        synapse_slow = NonSpikingSynapse(max_conductance=g_in, reversal_potential=reversal_in, e_lo=0.0, e_hi=activity_range)

    neuron_fast = NonSpikingNeuron(membrane_conductance=1.0, membrane_capacitance=calc_cap_from_cutoff(cutoff_higher), resting_potential=rest)
    neuron_slow = NonSpikingNeuron(membrane_conductance=1.0, membrane_capacitance=calc_cap_from_cutoff(cutoff_lower), resting_potential=rest)

    net.add_neuron(neuron_fast, name=name+'_in')
    net.add_neuron(neuron_fast, name=name+'_fast', initial_value=0.0)
    net.add_neuron(neuron_slow, name=name+'_slow', initial_value=0.0)
    net.add_neuron(neuron_fast, name=name+'_out')

    net.add_connection(synapse_fast, name+'_in', name+'_fast')
    net.add_connection(synapse_fast, name+'_in', name+'_slow')
    net.add_connection(synapse_fast, name+'_fast', name+'_out')
    net.add_connection(synapse_slow, name+'_slow', name+'_out')

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

    neuron_fast = NonSpikingNeuron(membrane_conductance=1.0, membrane_capacitance=calc_cap_from_cutoff(cutoff_fastest), resting_potential=rest)
    neuron_b = NonSpikingNeuron(membrane_conductance=1.0, membrane_capacitance=calc_cap_from_cutoff(cutoff_higher), resting_potential=rest)
    neuron_slow = NonSpikingNeuron(membrane_conductance=1.0, membrane_capacitance=calc_cap_from_cutoff(cutoff_lower), resting_potential=rest)

    net.add_neuron(neuron_fast, name=name+'_in')
    net.add_neuron(neuron_b, name=name+'_fast', initial_value=0.0)
    net.add_neuron(neuron_slow, name=name+'_slow', initial_value=0.0)
    net.add_neuron(neuron_fast, name=name+'_out')

    net.add_connection(synapse_fast, name+'_in', name+'_fast')
    net.add_connection(synapse_fast, name+'_in', name+'_slow')
    net.add_connection(synapse_bd, name+'_fast', name+'_out')
    net.add_connection(synapse_slow, name+'_slow', name+'_out')

c_fastest = calc_cap_from_cutoff(cutoff_fastest)
dt = c_fastest/10
