import pickle
import numpy as np

from frequency_response import sample_frequency_response

from neurons_and_networks import filter_lowpass
from sns_toolbox.neurons import NonSpikingNeuron
from sns_toolbox.connections import NonSpikingSynapse
from sns_toolbox.networks import Network
from sns_toolbox.renderer import render

dt = 0.01
cutoff_low = 5
cutoff_high = 500
R = 2.0

del_e_ex = 3*R
del_e_in = -1*R

g_add = R/(del_e_ex-R)
# g_sub = -g_add*del_e_ex/del_e_in
g_sub = 0.139
net = Network()

filter_lower = filter_lowpass(cutoff_low, dt, name='5 Hz')
filter_higher = filter_lowpass(cutoff_high, dt, name='500 Hz')

f_sample = 1000 / dt
w_cutoff = 2 * np.pi / f_sample * cutoff_high
alpha = 0.5*(1/np.cos(w_cutoff))*2*(-1+np.cos(w_cutoff)-np.sin(w_cutoff))
if alpha <= 0:
    alpha = 0.5 * (1 / np.cos(w_cutoff)) * 2 * (-1 + np.cos(w_cutoff) + np.sin(w_cutoff))
c_m = dt / alpha

neuron_type = NonSpikingNeuron(membrane_capacitance=c_m)
net.add_neuron(neuron_type, name='Bandpass')

net.add_network(filter_higher)
net.add_network(filter_lower)

net.add_neuron(neuron_type, name='Source')
net.add_input('Source')
net.add_output('Bandpass',spiking=False)

synapse_add = NonSpikingSynapse(max_conductance=g_add, reversal_potential=del_e_ex, e_hi=R)
synapse_sub = NonSpikingSynapse(max_conductance=g_sub, reversal_potential=del_e_in, e_hi=R)
net.add_connection(synapse_add,'500 Hz','Bandpass')
net.add_connection(synapse_sub,'5 Hz','Bandpass')
net.add_connection(synapse_add,'Source', '500 Hz')
net.add_connection(synapse_add,'Source', '5 Hz')

# render(net)

model = net.compile(backend='numpy',dt=dt)

frequencies, output_peaks, phase_diff_deg = sample_frequency_response(model, low_hz=0.1, high_hz=1000, num_samples=100, plot=False, debug=True)

data = {'frequencies': frequencies, 'outputPeaks': output_peaks, 'phaseDiff': phase_diff_deg, 'dt': dt, 'cutoffHigh': cutoff_high, 'cutoffLow': cutoff_low}

pickle.dump(data, open("data_bandpass.p", 'wb'))
