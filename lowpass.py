import sns_toolbox
import numpy as np
import matplotlib.pyplot as plt
from frequency_response import sample_frequency_response

from sns_toolbox.networks import Network
from sns_toolbox.neurons import NonSpikingNeuron
from neurons_and_networks import filter_lowpass

# dt = 0.1    # ms

# net = Network()
dt = 0.1
cutoff = 10
# sample_freq = 1000/dt
# w_cutoff = 2*np.pi/sample_freq*cutoff
# alpha = np.cos(w_cutoff) - 1 + np.sqrt((np.cos(w_cutoff))**2 - 4*np.cos(w_cutoff) + 3)
# cap = dt/alpha
# print(cap)
# neuron_type = NonSpikingNeuron(membrane_conductance=1.0, membrane_capacitance=cap)
# net.add_neuron(neuron_type)
net = filter_lowpass(cutoff, dt)
net.add_input(0)
net.add_output(0,spiking=False)

model = net.compile(backend='numpy',dt=dt)

fig, _, _ = sample_frequency_response(model, low_hz=0.1, high_hz=100, num_samples=100, plot=True, debug=False)

# # Calculated cutoff 1
# sample_freq = 1/(dt/1000)
# alpha = 1/cap
# calc1 = sample_freq/(2*np.pi)*np.arccos(1-(alpha**2)/(2*(1-alpha)))
# print(calc1)
#
# # Calculated cutoff 2
# alpha = dt/cap
# calc2 = 1/(2*np.pi)*np.arccos(1-(alpha**2)/(2*(1-alpha)))
# print(calc2)

# Calculated cutoff 3
# alpha = dt/cap
# calc3 = sample_freq/(2*np.pi)*np.arccos(1-(alpha**2)/(2*(1-alpha)))
# print(calc3)

# # Calcualted cutoff 4
# sample_freq = 1/(dt)
# alpha = 1/cap
# calc4 = sample_freq/(2*np.pi)*np.arccos(1-(alpha**2)/(2*(1-alpha)))
# print(calc4)
#
# # Calculated cutoff 5
# alpha = dt/cap
# calc5 = sample_freq/(2*np.pi)*np.arccos(1-(alpha**2)/(2*(1-alpha)))
# print(calc5)
#
# # Calculated cutoff 6
# alpha = 1/cap
# calc6 = alpha/((1-alpha)*2*np.pi*dt)
# print(calc6)

# plt.axvline(x=calc1, color='orange', label='Calc1')
# plt.axvline(x=calc2, color='green', label='Calc2')
plt.axvline(x=cutoff, color='red', label='Cutoff Freq')
# plt.axvline(x=calc4, color='blue', label='Calc4')
# plt.axvline(x=calc5, color='black', label='Calc5')
# plt.axvline(x=calc6, color='cyan', label='Calc6')
plt.legend()
plt.show()
