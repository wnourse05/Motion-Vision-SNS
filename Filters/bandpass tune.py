import pickle
import numpy as np

from neurons_and_networks import filter_lowpass
from sns_toolbox.neurons import NonSpikingNeuron
from sns_toolbox.connections import NonSpikingSynapse
from sns_toolbox.networks import Network
from sns_toolbox.renderer import render

dt = 0.01
cutoff_low = 5
cutoff_high = 500
R = 2.0

def gen_network(g_sub, dt, cutoff_low, cutoff_high, R):
    del_e_ex = 3 * R
    del_e_in = -1 * R

    g_add = R / (del_e_ex - R)

    net = Network()

    filter_lower = filter_lowpass(cutoff_low, dt, name='5 Hz')
    filter_higher = filter_lowpass(cutoff_high, dt, name='500 Hz')

    f_sample = 1000 / dt
    w_cutoff = 2 * np.pi / f_sample * cutoff_high
    alpha = 0.5 * (1 / np.cos(w_cutoff)) * 2 * (-1 + np.cos(w_cutoff) - np.sin(w_cutoff))
    if alpha <= 0:
        alpha = 0.5 * (1 / np.cos(w_cutoff)) * 2 * (-1 + np.cos(w_cutoff) + np.sin(w_cutoff))
    c_m = dt / alpha

    neuron_type = NonSpikingNeuron(membrane_capacitance=c_m)
    net.add_neuron(neuron_type, name='Bandpass')

    net.add_network(filter_higher)
    net.add_network(filter_lower)

    net.add_neuron(neuron_type, name='Source')
    net.add_input('Source')
    net.add_output('Bandpass', spiking=False)

    synapse_add = NonSpikingSynapse(max_conductance=g_add, reversal_potential=del_e_ex, e_hi=R)
    synapse_sub = NonSpikingSynapse(max_conductance=g_sub, reversal_potential=del_e_in, e_hi=R)
    net.add_connection(synapse_add, '500 Hz', 'Bandpass')
    net.add_connection(synapse_sub, '5 Hz', 'Bandpass')
    net.add_connection(synapse_add, 'Source', '500 Hz')
    net.add_connection(synapse_add, 'Source', '5 Hz')

    model = net.compile(backend='numpy', dt=dt)
    return model

def test_network_single_freq(model, freq):
    dt_s = model.dt / 1000
    t_max = (1 / freq) * 5
    t = np.arange(start=0.0, stop=t_max, step=dt_s)

    # inp = np.zeros(len(t)])
    inp = np.sin(2 * np.pi * freq * t) + 1

    outputs = np.zeros_like(inp)

    print('  Frequency ' + str(freq) + ' Hz')
    model.reset()
    for step in range(len(t)):
        if t[step] > (3 * (1 / freq)):
            break
        outputs[step] = model([inp[step]])

    output_peak = np.max(outputs[int(step / 2):step])

    return output_peak

def collect_data(g_sub, dt, cutoff_low, cutoff_high, R):
    num_samples = len(g_sub)
    peaks_low = np.zeros_like(g_sub)
    peaks_high = np.zeros_like(g_sub)
    for i in range(num_samples):
        print('Sample ' + str(i+1) + '/' + str(num_samples))
        model = gen_network(g_sub[i], dt, cutoff_low, cutoff_high, R)
        peaks_low[i] = test_network_single_freq(model, cutoff_low)
        peaks_high[i] = test_network_single_freq(model, cutoff_high)
    return peaks_low, peaks_high


g_sub = np.linspace(0.00001,1.0, num=50)

peaks_low, peaks_high = collect_data(g_sub, dt, cutoff_low, cutoff_high, R)

data = {'peaks_low': peaks_low, 'freq_low': cutoff_low, 'peaks_high': peaks_high, 'freq_high': cutoff_high, 'g_sub': g_sub}

pickle.dump(data, open("data_bandpass_test.p", 'wb'))
