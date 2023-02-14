import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from sns_toolbox.neurons import NonSpikingNeuron
from sns_toolbox.connections import NonSpikingSynapse
from sns_toolbox.networks import Network
from sns_toolbox.renderer import render
from utilities import lowpass_filter, bandpass_filter, synapse_target, calc_cap_from_cutoff, activity_range

"""
########################################################################################################################
NETWORK
"""

cutoff_fastest = 200    # hz
c_fastest = calc_cap_from_cutoff(cutoff_fastest)
dt = c_fastest/10

net = Network('Motion Vision Single Column')

"""
########################################################################################################################
RETINA
"""

retina = lowpass_filter(net, cutoff_fastest, name='Retina')

net.add_input('Retina')
net.add_output('Retina', name='OutR')

"""
########################################################################################################################
LAMINA
"""
g_r_l, reversal_r_l = synapse_target(0.0, activity_range)
synapse_r_l = NonSpikingSynapse(max_conductance=g_r_l, reversal_potential=reversal_r_l, e_lo=0.0, e_hi=activity_range)

l1 = bandpass_filter(net, cutoff_lower=cutoff_fastest/10, cutoff_higher=cutoff_fastest, name='L1', invert=True)
l2 = bandpass_filter(net, cutoff_lower=cutoff_fastest/5, cutoff_higher=cutoff_fastest, name='L2', invert=True)
l3 = lowpass_filter(net, cutoff=cutoff_fastest, name='L3', invert=True)
l5 = lowpass_filter(net, cutoff=cutoff_fastest, name='L5', invert=False, bias=activity_range)

net.add_connection(synapse_r_l, 'Retina', 'L1_in')
net.add_connection(synapse_r_l, 'Retina', 'L2_in')
net.add_connection(synapse_r_l, 'Retina', 'L3')
net.add_connection(synapse_r_l, 'L1_out', 'L5')

net.add_output('L1_out', name='OutL1')
net.add_output('L2_out', name='OutL2')
net.add_output('L3', name='OutL3')
net.add_output('L5', name='OutL5')

"""
########################################################################################################################
SIMULATE
"""
# render(net, view=True)
model = net.compile(dt, backend='torch', device='cpu')

t = np.arange(0,50, dt)
inputs = torch.ones([len(t), net.get_num_inputs()])
data = np.zeros([len(t), net.get_num_outputs_actual()])

for i in range(len(t)):
    data[i,:] = model(inputs[i,:])
data = data.transpose()

"""
########################################################################################################################
PLOTTING
"""
group = True

if group:
    plt.figure()
    grid = matplotlib.gridspec.GridSpec(2, 8)

if group:
    plt.subplot(grid[0,0])
else:
    plt.figure()
plt.plot(t, data[:][0])
plt.ylim([-0.1,1.1])
plt.title('Retina')

if group:
    plt.subplot(grid[1,0])
else:
    plt.figure()
plt.plot(t, data[:][1])
plt.ylim([-0.1,1.1])
plt.title('L1')

if group:
    plt.subplot(grid[1,1])
else:
    plt.figure()
plt.plot(t, data[:][2])
plt.ylim([-0.1,1.1])
plt.title('L2')

if group:
    plt.subplot(grid[1,2])
else:
    plt.figure()
plt.plot(t, data[:][3])
plt.ylim([-0.1,1.1])
plt.title('L3')

if group:
    plt.subplot(grid[1,3])
else:
    plt.figure()
plt.plot(t, data[:][4])
plt.ylim([-0.1,1.1])
plt.title('L5')

plt.show()
