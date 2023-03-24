import pickle
import blosc
import numpy as np
import torch
from typing import Any
import numbers
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    pickled = pickle.dumps(data)
    compressed = blosc.compress(pickled)
    with open(filename, 'wb') as f:
        f.write(compressed)

def load_data(filename):
    with open(filename, 'rb') as f:
        compressed = f.read()
    decompressed = blosc.decompress(compressed)
    data = pickle.loads(decompressed)
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

c_fastest = calc_cap_from_cutoff(cutoff_fastest)
dt = c_fastest/10

def gen_gratings(wavelength, angle, vel, dt, num_steps, fov=160, res=5):
    # Generate meshgrid
    x = np.arange(0, fov, res)
    x_rad = np.deg2rad(x)
    X, Y = np.meshgrid(x_rad, x_rad)
    Y = np.flipud(Y)

    wavelength_rad = np.deg2rad(wavelength)
    angle_rad = np.deg2rad(angle)

    dt_s = dt / 1000
    disp = vel * dt_s
    disp_rad = np.deg2rad(disp)

    for i in range(num_steps):
        grating = 0.5 * np.sin(2 * np.pi * (X * np.cos(angle_rad) + Y * np.sin(angle_rad)) / wavelength_rad + i*disp_rad) + 0.5
        grating_flat = grating.flatten()
        if i == 0:
            gratings = np.copy(grating_flat)
        else:
            gratings = np.vstack((gratings, grating_flat))

    return gratings

