from sns_toolbox.networks import Network
from sns_toolbox.neurons import NonSpikingNeuron

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