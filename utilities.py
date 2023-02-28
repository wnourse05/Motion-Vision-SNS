import pickle
import numpy as np
import torch
from typing import Any
import numbers
import matplotlib.pyplot as plt

from sns_toolbox.neurons import NonSpikingNeuron
from sns_toolbox.connections import NonSpikingSynapse, NonSpikingMatrixConnection
from sns_toolbox.networks import Network

c_fast = 1.0
activity_range = 1.0
reversal_ex = 5.0
reversal_in = -2.0
backend = 'torch'
device = 'cuda'
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

def single_to_diagonal(val, size):
    val_array = np.zeros(size) + val
    val_matrix = np.diag(val_array)
    return val_matrix
class NonSpikingOneToOneConnection(NonSpikingMatrixConnection):
    """
    :param max_conductance: Synaptic conductance in uS
    :type max_conductance: Number
    :param reversal_potential: Reversal potential in mV
    :type reversal_potential: Number
    :param e_lo: Synaptic activation threshold in mV
    :type e_lo: Number
    :param e_hi: Synaptic maximum activation limit in mV
    :type e_hi: Number
    """
    def __init__(self, shape, max_conductance, reversal_potential, e_lo, e_hi, **kwargs: Any) -> None:
        if len(shape) > 1:
            size = int(shape[0]*shape[1])
        else:
            size = shape[0]
        conductance_matrix = single_to_diagonal(max_conductance, size)
        reversal_matrix = single_to_diagonal(reversal_potential, size)
        e_lo_matrix = np.zeros([size, size]) + e_lo
        e_hi_matrix = np.zeros([size, size]) + e_hi
        super().__init__(conductance_matrix, reversal_matrix, e_lo_matrix, e_hi_matrix, **kwargs)

def add_lowpass_filter(net: Network, cutoff, shape=None, invert=False, name=None, bias=0.0, initial_value=0.0, **kwargs):
    membrane_conductance = 1.0
    membrane_capacitance = calc_cap_from_cutoff(cutoff)
    if invert:
        rest = activity_range
    else:
        rest = 0.0
    if name is None:
        name = 'lowpass'
    if shape is None:
        shape = [1]
    neuron_type = NonSpikingNeuron(membrane_conductance=membrane_conductance, membrane_capacitance=membrane_capacitance, resting_potential=rest, bias=bias, **kwargs)
    net.add_population(neuron_type, name=name, initial_value=initial_value, shape=shape)

def add_scaled_bandpass_filter(net: Network, cutoff_lower, cutoff_higher, k, invert=False, name=None, shape=None, **kwargs):
    if shape is None:
        shape = [1]
    if name is None:
        name = 'bandpass'
    if invert:
        rest = activity_range
        g_in = (-activity_range)/reversal_in
        g_bd = (-k*activity_range)/(reversal_in + k*activity_range)
        g_cd = (g_bd*(reversal_in-activity_range))/(activity_range-reversal_ex)
        # if shape == [1]:
        #     synapse_fast = NonSpikingSynapse(max_conductance=g_in, reversal_potential=reversal_in, e_lo=0.0, e_hi=activity_range)
        #     synapse_bd = NonSpikingSynapse(max_conductance=g_bd, reversal_potential=reversal_in, e_lo=0.0, e_hi=activity_range)
        #     synapse_slow = NonSpikingSynapse(max_conductance=g_cd, reversal_potential=reversal_ex, e_lo=0.0, e_hi=activity_range)
        # else:
        synapse_fast = NonSpikingOneToOneConnection(shape=shape, max_conductance=g_in, reversal_potential=reversal_in, e_lo=0.0,
                                         e_hi=activity_range)
        synapse_bd = NonSpikingOneToOneConnection(shape=shape, max_conductance=g_bd, reversal_potential=reversal_in, e_lo=0.0,
                                         e_hi=activity_range)
        synapse_slow = NonSpikingOneToOneConnection(shape=shape, max_conductance=g_cd, reversal_potential=reversal_ex, e_lo=0.0,
                                         e_hi=activity_range)
    else:
        rest = 0.0
        g_ex = activity_range/(reversal_ex-activity_range)
        g_bd = k*activity_range/(reversal_ex-k*activity_range)
        g_cd = (-g_bd*reversal_ex)/reversal_in
        # if shape == [1]:
        #     synapse_fast = NonSpikingSynapse(max_conductance=g_ex, reversal_potential=reversal_ex, e_lo=0.0, e_hi=activity_range)
        #     synapse_bd = NonSpikingSynapse(max_conductance=g_bd, reversal_potential=reversal_ex, e_lo=0.0, e_hi=activity_range)
        #     synapse_slow = NonSpikingSynapse(max_conductance=g_cd, reversal_potential=reversal_in, e_lo=0.0, e_hi=activity_range)
        # else:
        synapse_fast = NonSpikingOneToOneConnection(shape=shape, max_conductance=g_ex,
                                                    reversal_potential=reversal_ex, e_lo=0.0,
                                                    e_hi=activity_range)
        synapse_bd = NonSpikingOneToOneConnection(shape=shape, max_conductance=g_bd, reversal_potential=reversal_ex,
                                                  e_lo=0.0,
                                                  e_hi=activity_range)
        synapse_slow = NonSpikingOneToOneConnection(shape=shape, max_conductance=g_cd,
                                                    reversal_potential=reversal_in, e_lo=0.0,
                                                    e_hi=activity_range)

    neuron_fast = NonSpikingNeuron(membrane_conductance=1.0, membrane_capacitance=calc_cap_from_cutoff(cutoff_fastest), resting_potential=rest, **kwargs)
    neuron_b = NonSpikingNeuron(membrane_conductance=1.0, membrane_capacitance=calc_cap_from_cutoff(cutoff_higher), resting_potential=rest, **kwargs)
    neuron_slow = NonSpikingNeuron(membrane_conductance=1.0, membrane_capacitance=calc_cap_from_cutoff(cutoff_lower), resting_potential=rest, **kwargs)

    net.add_population(neuron_fast, shape, name=name+'_in')
    net.add_population(neuron_b, shape, name=name+'_fast', initial_value=0.0)
    net.add_population(neuron_slow, shape, name=name+'_slow', initial_value=0.0)
    net.add_population(neuron_fast, shape, name=name+'_out')

    net.add_connection(synapse_fast, name+'_in', name+'_fast')
    net.add_connection(synapse_fast, name+'_in', name+'_slow')
    net.add_connection(synapse_bd, name+'_fast', name+'_out')
    net.add_connection(synapse_slow, name+'_slow', name+'_out')

class Stimulus:
    def __init__(self, stimulus_array, interval):
        self.stimulus_array = stimulus_array
        self.interval = interval
        self.index = 0
        self.interval_ctr = -1
        self.num_rows = np.shape(stimulus_array)[0]

    def get_stimulus(self):
        self.interval_ctr += 1
        if self.interval_ctr >= self.interval:
            self.interval_ctr = 0
            self.index += 1
            if self.index >= self.num_rows:
                self.index = 0

        return self.stimulus_array[self.index, :]

c_fastest = calc_cap_from_cutoff(cutoff_fastest)
dt = c_fastest/10

def gen_gratings(shape, freq, dir, num_cycles, square=False):
    # Get the number of samples
    period = 1/freq
    num_samples = int(num_cycles*period/(dt/1000))
    num_samples_period = int(period/(dt/1000))

    # Generate the reference wave
    if square:
        x = torch.zeros(num_samples_period, device=device)
        x[:int(num_samples_period / 2)] = 1.0
        y = torch.zeros(num_samples_period, device=device)
        y[:] = x
    else:
        x = torch.linspace(0, num_cycles*2*np.pi, num_samples, device=device)
        y = torch.sin(x)/2+0.5

    for i in range(num_cycles):
        y = torch.hstack((y,x))
    if dir == 'a':
        start_at_end = False
        up_down = False
    elif dir == 'b':
        start_at_end = True
        up_down = False
    elif dir == 'c':
        start_at_end = False
        up_down = True
    elif dir == 'd':
        start_at_end = True
        up_down = True
    else:
        raise ValueError('Invalid direction, must be a b c or d')

    # Construct the matrices
    num_rows = shape[0]
    num_cols = shape[1]
    if up_down:
        window_length = num_rows
    else:
        window_length = num_cols
    matrix = torch.zeros(shape, device=device)
    stimulus = torch.zeros(num_cols*num_rows, device=device)

    # Fill the matrices
    for i in range(num_samples-window_length+1):
        window = y[i:i+window_length]
        if up_down:
            for col in range(num_cols):
                matrix[:,col] = window
        else:
            for row in range(num_rows):
                matrix[row,:] = window
        if i == 0:
            stimulus = torch.flatten(matrix)
        else:
            if start_at_end:
                stimulus = torch.vstack((torch.flatten(matrix), stimulus))
            else:
                stimulus = torch.vstack((stimulus, torch.flatten(matrix)))

    return stimulus, y
