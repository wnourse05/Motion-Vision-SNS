from sns_toolbox.networks import Network
from sns_toolbox.neurons import NonSpikingNeuron
from sns_toolbox.connections import NonSpikingTransmissionSynapse, NonSpikingSynapse

import numpy as np

def filter_lowpass(f_cutoff, dt, resting_potential=0.0, name=None, bias=0.0, color=None, initial_value=None):
    """
    Generate a neuron which acts as a lowpass filter with the specified cutoff frequency.
    :param f_cutoff:    Desired cutoff frequency (Hz)
    :type f_cutoff:     Number
    :param dt:          Simulation timestep (ms)
    :type dt:           Number
    :return:            Lowpass filter network
    :rtype:             sns_toolbox.networks.Network
    """
    if name is None:
        name = 'Lowpass Filter'
    f_sample = 1000/dt
    w_cutoff = 2*np.pi/f_sample*f_cutoff
    alpha = np.cos(w_cutoff) - 1 + np.sqrt((np.cos(w_cutoff)) ** 2 - 4 * np.cos(w_cutoff) + 3)
    c_m = dt / alpha

    neuron_type = NonSpikingNeuron(membrane_conductance=1.0, membrane_capacitance=c_m, resting_potential=resting_potential, bias=bias)

    net = Network(name=name)
    net.add_neuron(neuron_type, name=name, color=color, initial_value=initial_value)

    return net

def filter_highpass(f_cutoff, dt, pass_capacitance=1, R=20.0, resting_potential=0.0, name=None, bias=0.0, color=None, initial_value=None, add_del_e=100, sub_del_e=-40):

    if name is None:
        name = 'Highpass Filter'
    f_sample = 1000 / dt
    w_cutoff = 2 * np.pi / f_sample * f_cutoff
    alpha = 0.5*(1/np.cos(w_cutoff))*2*(-1+np.cos(w_cutoff)-np.sin(w_cutoff))
    if alpha <= 0:
        alpha = 0.5 * (1 / np.cos(w_cutoff)) * 2 * (-1 + np.cos(w_cutoff) + np.sin(w_cutoff))
    c_m = dt / alpha
    gain = -(1-alpha)/alpha
    g_ex = R/(add_del_e-R)
    g_in = (-g_ex*add_del_e)*gain/sub_del_e

    neuron_type = NonSpikingNeuron(membrane_conductance=1.0, membrane_capacitance=c_m,
                                   resting_potential=resting_potential, bias=bias)

    net = Network(name=name)
    net.add_neuron(neuron_type, name=name, color=color, initial_value=initial_value)

    net.add_neuron(NonSpikingNeuron(membrane_capacitance=pass_capacitance), name='Highpass Input')  # 1
    net.add_neuron(NonSpikingNeuron(membrane_capacitance=pass_capacitance), name='Highpass Pass')   # 2
    net.add_neuron(NonSpikingNeuron(membrane_capacitance=pass_capacitance), name='Highpass Output') # 3

    synapse_add = NonSpikingTransmissionSynapse(gain=1, reversal_potential=add_del_e, e_lo=resting_potential, e_hi=resting_potential+R)
    synapse_sub = NonSpikingSynapse(max_conductance=g_in,reversal_potential=sub_del_e, e_lo=resting_potential, e_hi=resting_potential+R)

    net.add_connection(synapse_add, 1, 2)
    net.add_connection(synapse_add, 1, 0)
    net.add_connection(synapse_add, 2, 3)
    net.add_connection(synapse_sub, 0, 3)

    return net
