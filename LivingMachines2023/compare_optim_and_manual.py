from utilities import save_data, c_fast, activity_range, reversal_ex, reversal_in, dt, backend
import numpy as np
from sns_toolbox.neurons import NonSpikingNeuron
from sns_toolbox.connections import NonSpikingSynapse
from sns_toolbox.networks import Network
from sns_toolbox.renderer import render
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import minimize_scalar
from timeit import default_timer as timer

target_center = 0.0
target_middle = 7/8
target_outer = 1.0

target_field = np.array([[target_middle, target_middle, target_middle],
                         [target_middle, target_center, target_middle],
                         [target_middle, target_middle, target_middle]])

c_slow = 10 * c_fast
def create_bandpass_w_input(conductance_retina):
    net = Network()
    neuron_fast = NonSpikingNeuron(membrane_capacitance=c_fast, membrane_conductance=1.0, bias=activity_range)
    neuron_slow = NonSpikingNeuron(membrane_capacitance=c_slow, membrane_conductance=1.0, bias=activity_range)
    neuron_retina = NonSpikingNeuron(membrane_capacitance=c_fast, membrane_conductance=1.0)

    conductance_in = -activity_range/reversal_in
    conductance_ex = conductance_in*(reversal_in-activity_range)/(activity_range-reversal_ex)
    synapse_in = NonSpikingSynapse(max_conductance=conductance_in, reversal_potential=reversal_in, e_lo=0.0, e_hi=activity_range)
    synapse_ex = NonSpikingSynapse(max_conductance=conductance_ex, reversal_potential=reversal_ex, e_lo=0.0, e_hi=activity_range)

    synapse_retina = NonSpikingSynapse(max_conductance=conductance_retina, reversal_potential=reversal_in, e_lo=0.0, e_hi=activity_range)

    net.add_neuron(neuron_retina, name='Retina')
    net.add_neuron(neuron_fast, name='a', initial_value=activity_range)
    net.add_neuron(neuron_fast, name='b')
    net.add_neuron(neuron_slow, name='c')
    net.add_neuron(neuron_fast, name='d', initial_value=activity_range)

    net.add_connection(synapse_retina, 'Retina', 'a')
    net.add_connection(synapse_in, 'a', 'b')
    net.add_connection(synapse_in, 'a', 'c')
    net.add_connection(synapse_in, 'b', 'd')
    net.add_connection(synapse_ex, 'c', 'd')

    net.add_input('Retina')
    net.add_output('d')

    model = net.compile(dt=dt, backend=backend)

    return model

def run_net(conductance_retina):
    model = create_bandpass_w_input(conductance_retina)

    t = np.arange(0, 30, dt)
    data = np.zeros_like(t)

    for i in range(len(t)):
        data[i] = model([activity_range])

    # plt.plot(t, data)
    peak = np.min(data)

    return peak

def error(conductance_retina, target_peak):
    peak = run_net(conductance_retina)
    peak_error = (peak - target_peak)**2
    return peak_error

def tune_net(target_peak):
    f = lambda x : error(x, target_peak)
    res = minimize_scalar(f, bounds=(0.0,1.0), method='bounded')
    peak_error = res.fun
    conductance = res.x
    peak = run_net(conductance)
    return conductance, peak, np.sqrt(peak_error)

def synapse_manual(target):
    conductance = (activity_range - target)/(target - reversal_in)
    peak = run_net(conductance)
    peak_error = error(conductance, target)
    return conductance, peak, peak_error

# plt.figure()
# plt.title(str(target_middle))
start = timer()
conductance_middle, peak_middle, error_middle = tune_net(target_middle)
time_middle_opt = timer()
conductance_middle_manual, peak_middle_manual, error_middle_manual = synapse_manual(target_middle)
time_middle_manual = timer()
# plt.figure()
# plt.title(str(target_center))
conductance_center, peak_center, error_center = tune_net(target_center)
time_center_opt = timer()
conductance_center_manual, peak_center_manual, error_center_manual = synapse_manual(target_center)
time_center_manual = timer()

print('Average Optimization Time: ', (time_middle_opt-start + time_center_opt-time_middle_manual)/2)
print('Average Manual Time: ', (time_middle_manual - time_middle_opt + time_center_manual-time_center_opt)/2)

field = np.array([[peak_middle, peak_middle, peak_middle],
                  [peak_middle, peak_center, peak_middle],
                  [peak_middle, peak_middle, peak_middle]])

field_error = np.array([[error_middle, error_middle, error_middle],
                        [error_middle, error_center, error_middle],
                        [error_middle, error_middle, error_middle]])

field_manual = np.array([[peak_middle_manual, peak_middle_manual, peak_middle_manual],
                         [peak_middle_manual, peak_center_manual, peak_middle_manual],
                         [peak_middle_manual, peak_middle_manual, peak_middle_manual]])

field_error_manual = np.array([[error_middle_manual, error_middle_manual, error_middle_manual],
                               [error_middle_manual, error_center_manual, error_middle_manual],
                               [error_middle_manual, error_middle_manual, error_middle_manual]])

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("",["darkblue","blue","white","red","darkred"])
norm = plt.Normalize(0,2)

plt.figure()
plt.subplot(2,3,1)
plt.imshow(target_field, cmap=cmap, norm=norm)
plt.subplot(2,3,2)
plt.imshow(field, cmap=cmap, norm=norm)
plt.subplot(2,3,3)
plt.imshow(field_error)
plt.colorbar()
plt.subplot(2,3,5)
plt.imshow(field_manual, cmap=cmap, norm=norm)
plt.subplot(2,3,6)
plt.imshow(field_error_manual)
plt.colorbar()

plt.show()
