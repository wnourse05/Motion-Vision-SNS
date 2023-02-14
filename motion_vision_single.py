import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib

from sns_toolbox.neurons import NonSpikingNeuron
from sns_toolbox.connections import NonSpikingSynapse
from sns_toolbox.networks import Network
from utilities import add_lowpass_filter, synapse_target, activity_range, reversal_ex, reversal_in, cutoff_fastest, dt, load_data, add_scaled_bandpass_filter

"""
########################################################################################################################
NETWORK
"""

net = Network('Motion Vision Single Column')

"""
########################################################################################################################
RETINA
"""

params_node_retina = load_data('params_node_retina.p')
add_lowpass_filter(net, params_node_retina['params']['cutoff'], name='Retina', invert=params_node_retina['params']['invert'], initial_value=params_node_retina['params']['initialValue'], bias=params_node_retina['params']['bias'])

net.add_input('Retina')
net.add_output('Retina', name='OutR')

"""
########################################################################################################################
LAMINA
"""
g_r_l1, reversal_r_l1 = synapse_target(0.0, activity_range)
g_r_l2 = g_r_l1
reversal_r_l2 = reversal_r_l1
g_r_l3 = g_r_l1
reversal_r_l3 = reversal_r_l1

synapse_r_l1 = NonSpikingSynapse(max_conductance=g_r_l1, reversal_potential=reversal_r_l1, e_lo=0.0, e_hi=activity_range)
synapse_r_l2 = NonSpikingSynapse(max_conductance=g_r_l2, reversal_potential=reversal_r_l2, e_lo=0.0, e_hi=activity_range)
synapse_r_l3 = NonSpikingSynapse(max_conductance=g_r_l3, reversal_potential=reversal_r_l3, e_lo=0.0, e_hi=activity_range)

params_node_l1 = load_data('params_node_l1.p')
add_scaled_bandpass_filter(net, params_node_l1['params']['cutoffLow'], params_node_l1['params']['cutoffHigh'],
                           params_node_l1['params']['gain'], invert=params_node_l1['params']['invert'], name='L1')
params_node_l2 = load_data('params_node_l2.p')
add_scaled_bandpass_filter(net, params_node_l2['params']['cutoffLow'], params_node_l2['params']['cutoffHigh'],
                           params_node_l2['params']['gain'], invert=params_node_l2['params']['invert'], name='L2')
params_node_l3 = load_data('params_node_l3.p')
l3 = add_lowpass_filter(net, cutoff=params_node_l3['params']['cutoff'], name='L3', invert=params_node_l3['params']['invert'], initial_value=params_node_l3['params']['initialValue'], bias=params_node_l3['params']['bias'])
# l5 = lowpass_filter(net, cutoff=cutoff_fastest, name='L5', invert=False, bias=activity_range)

net.add_connection(synapse_r_l1, 'Retina', 'L1_in')
net.add_connection(synapse_r_l2, 'Retina', 'L2_in')
net.add_connection(synapse_r_l3, 'Retina', 'L3')
# net.add_connection(synapse_r_l, 'L1_out', 'L5')

net.add_output('L1_out', name='OutL1')
net.add_output('L2_out', name='OutL2')
net.add_output('L3', name='OutL3')
# net.add_output('L5', name='OutL5')

"""
########################################################################################################################
MEDULLA ON
"""
reversal_l3_mi9 = reversal_ex
g_l3_mi9 = activity_range/(reversal_l3_mi9 - activity_range)
g_l1_mi1, reversal_l1_mi1 = synapse_target(0.0, activity_range)

synapse_l3_mi9 = NonSpikingSynapse(max_conductance=g_l3_mi9, reversal_potential=reversal_l3_mi9, e_lo=0.0, e_hi=activity_range)

params_node_mi1 = load_data('params_node_mi1.p')
mi1 = add_lowpass_filter(net, cutoff=cutoff_fastest, name='Mi1', invert=False, bias=activity_range)
mi9 = add_lowpass_filter(net, cutoff=cutoff_fastest, name='Mi9', invert=False, initial_value=activity_range)

params_conn_mi1 = load_data('params_conn_mi1.p')
synapse_l1_mi1 = NonSpikingSynapse(max_conductance=0.5, reversal_potential=params_conn_mi1['reversal'], e_lo=0.0, e_hi=activity_range)
net.add_connection(synapse_l1_mi1, 'L1_out', 'Mi1')
net.add_connection(synapse_l3_mi9, 'L3', 'Mi9')

net.add_output('Mi1', name='OutMi1')
net.add_output('Mi9', name='OutMi9')

"""
########################################################################################################################
MEDULLA OFF
"""
reversal_l3_tm9 = reversal_ex
g_l3_tm9 = activity_range/(reversal_l3_mi9 - activity_range)
reversal_l2_tm1 = reversal_ex
g_l2_tm1 = activity_range/(reversal_l2_tm1 - activity_range)

synapse_l2_tm1 = NonSpikingSynapse(max_conductance=g_l2_tm1, reversal_potential=reversal_l2_tm1, e_lo=0.0, e_hi=activity_range)
synapse_l3_tm9 = NonSpikingSynapse(max_conductance=g_l3_tm9, reversal_potential=reversal_l3_tm9, e_lo=0.0, e_hi=activity_range)

tm1 = add_lowpass_filter(net, cutoff=cutoff_fastest, name='Tm1', invert=False, initial_value=activity_range)
tm9 = add_lowpass_filter(net, cutoff=cutoff_fastest, name='Tm9', invert=False, initial_value=activity_range)

net.add_connection(synapse_l2_tm1, 'L2_out', 'Tm1')
net.add_connection(synapse_l3_mi9, 'L3', 'Tm9')

net.add_output('Tm1', name='OutTm1')
net.add_output('Tm9', name='OutTm9')

"""
########################################################################################################################
CT1 COMPARTMENTS
"""
reversal_mi1_ct1on = reversal_ex
g_mi1_ct1on = activity_range/(reversal_mi1_ct1on - activity_range)
reversal_tm1_ct1off = reversal_ex
g_tm1_ct1off = activity_range/(reversal_tm1_ct1off - activity_range)

synapse_mi1_ct1on = NonSpikingSynapse(max_conductance=g_mi1_ct1on, reversal_potential=reversal_mi1_ct1on, e_lo=0.0, e_hi=activity_range)
synapse_tm1_ct1off = NonSpikingSynapse(max_conductance=g_tm1_ct1off, reversal_potential=reversal_tm1_ct1off, e_lo=0.0, e_hi=activity_range)

ct1_on = add_lowpass_filter(net, cutoff=cutoff_fastest, name='CT1_On', invert=False)
ct1_off = add_lowpass_filter(net, cutoff=cutoff_fastest, name='CT1_Off', invert=False, initial_value=activity_range)

net.add_connection(synapse_mi1_ct1on, 'Mi1', 'CT1_On')
net.add_connection(synapse_tm1_ct1off, 'Tm1', 'CT1_Off')

net.add_output('CT1_On', name='OutCT1On')
net.add_output('CT1_Off', name='OutCT1Off')

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
    grid = matplotlib.gridspec.GridSpec(4, 4)

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
    plt.subplot(grid[2,0])
else:
    plt.figure()
plt.plot(t, data[:][4])
plt.ylim([-0.1,1.1])
plt.title('Mi1')

if group:
    plt.subplot(grid[2,1])
else:
    plt.figure()
plt.plot(t, data[:][5])
plt.ylim([-0.1,1.1])
plt.title('Mi9')

if group:
    plt.subplot(grid[2,2])
else:
    plt.figure()
plt.plot(t, data[:][6])
plt.ylim([-0.1,1.1])
plt.title('Tm1')

if group:
    plt.subplot(grid[2,3])
else:
    plt.figure()
plt.plot(t, data[:][7])
plt.ylim([-0.1,1.1])
plt.title('Tm9')

if group:
    plt.subplot(grid[3,0])
else:
    plt.figure()
plt.plot(t, data[:][8])
plt.ylim([-0.1,1.1])
plt.title('CT1_On')

if group:
    plt.subplot(grid[3,1])
else:
    plt.figure()
plt.plot(t, data[:][9])
plt.ylim([-0.1,1.1])
plt.title('CT1_Off')

plt.show()
