"""
Display the behavior of each neuron in the network
"""
import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import signal
import scipy.optimize as optimize
from sns_toolbox.networks import Network
from sns_toolbox.neurons import NonSpikingNeuron
from sns_toolbox.connections import NonSpikingSynapse

"""
########################################################################################################################
Helper Functions
"""
def calc_1d_point(x, A_rel, std_cen, std_sur):
    return np.exp((-x**2)/(2*std_cen**2)) - A_rel*np.exp((-x**2)/(2*std_sur**2))

def calc_2d_point(x, y, A_rel, std_cen, std_sur):
    return np.exp((-(x**2 + y**2))/(2*std_cen**2)) - A_rel*np.exp((-(x**2 + y**2))/(2*std_sur**2))

def adjust_spatiotemporal_data(t, data, index, R, rest=0.0, res=5, min_angle=-10, max_angle=10):
    # Original Receptive Field
    axis = np.arange(min_angle, max_angle + res, res)
    field_1d = np.zeros(len(axis))
    for i in range(len(axis)):
        field_1d[i] = data['polarity'][index] * calc_1d_point(axis[i], data['A_rel'][index], data['std_cen'][index],
                                                               data['std_sur'][index])
    field_2d = np.zeros([len(axis), len(axis)])
    for i in range(len(axis)):
        for j in range(len(axis)):
            field_2d[i, j] = data['polarity'][index] * calc_2d_point(axis[i], axis[j], data['A_rel'][index],
                                                                      data['std_cen'][index], data['std_sur'][index])

    # Scaled Receptive Field
    max_val = np.max(field_1d)
    min_val = np.min(field_1d)
    diff = max_val - min_val
    field_1d_norm = field_1d/diff
    field_1d_scaled = field_1d_norm*R
    new_rest = rest-np.min(field_1d_scaled)
    field_1d_shifted = field_1d_scaled + new_rest

    field_2d_norm = field_2d / diff
    field_2d_scaled = field_2d_norm * R
    field_2d_shifted = field_2d_scaled + new_rest

    # Original Step Response
    lowcut = 1 / (data['tau_lp'][index])
    if data['tau_hp'][index] is None:
        # print(lowcut)
        b, a = signal.butter(1, lowcut, 'low', analog=True)
    else:
        highcut = 1 / (data['tau_hp'][index])
        # print(lowcut, highcut)
        b, a = signal.butter(1, [highcut, lowcut], 'band', analog=True)
    _, y = signal.step((b, a), T=t)

    # Scaled Step Response
    polarity = data['polarity'][index]
    if polarity > 0:
        peak = np.max(field_1d_shifted)
        peak_y = np.max(y*polarity)
    else:
        peak = np.min(field_1d_shifted)
        peak_y = np.min(y*polarity)
    peak_diff = np.abs(peak-new_rest)
    y_norm = y/np.abs(peak_y)
    y_scaled = y_norm * peak_diff * polarity
    y_shifted = y_scaled + new_rest
    # print(np.min(y_shifted), np.max(y_shifted))
    # print(rest-np.min(field_1d_scaled))

    return axis, field_2d, field_2d_shifted, field_1d, field_1d_shifted, y*data['polarity'][index], y_shifted

def construct_bandpass(params_neuron, original_data, original_index, dt):
    g_ab = params_neuron['g_ab']
    g_ac = params_neuron['g_ac']
    g_bd = params_neuron['g_bd']
    g_cd = params_neuron['g_cd']

    del_e_ab = params_neuron['del_e_ab']
    del_e_ac = params_neuron['del_e_ac']
    del_e_bd = params_neuron['del_e_bd']
    del_e_cd = params_neuron['del_e_cd']

    c_a = params_neuron['c_fast']
    c_b = params_neuron['c_fast']
    c_c = params_neuron['c_slow']
    c_d = params_neuron['c_fast']

    rest = (-0.5*original_data['polarity'][original_index]+0.5)

    net = Network()

    neuron_a = NonSpikingNeuron(membrane_capacitance=c_a, membrane_conductance=1.0, resting_potential=rest)
    neuron_b = NonSpikingNeuron(membrane_capacitance=c_b, membrane_conductance=1.0, resting_potential=rest)
    neuron_c = NonSpikingNeuron(membrane_capacitance=c_c, membrane_conductance=1.0, resting_potential=rest)
    neuron_d = NonSpikingNeuron(membrane_capacitance=c_d, membrane_conductance=1.0, resting_potential=rest)

    synapse_ab = NonSpikingSynapse(max_conductance=g_ab, reversal_potential=del_e_ab, e_lo=0.0, e_hi=1.0)
    synapse_ac = NonSpikingSynapse(max_conductance=g_ac, reversal_potential=del_e_ac, e_lo=0.0, e_hi=1.0)
    synapse_bd = NonSpikingSynapse(max_conductance=g_bd, reversal_potential=del_e_bd, e_lo=0.0, e_hi=1.0)
    synapse_cd = NonSpikingSynapse(max_conductance=g_cd, reversal_potential=del_e_cd, e_lo=0.0, e_hi=1.0)

    net.add_neuron(neuron_a)
    net.add_neuron(neuron_b, initial_value=0.0)
    net.add_neuron(neuron_c, initial_value=0.0)
    net.add_neuron(neuron_d)

    net.add_connection(synapse_ab,0,1)
    net.add_connection(synapse_ac,0,2)
    net.add_connection(synapse_bd,1,3)
    net.add_connection(synapse_cd,2,3)

    net.add_input(0)
    net.add_output(3)

    model = net.compile(dt)

    return model

def construct_lowpass(params_neuron, dt):
    c = params_neuron['c']
    rest = params_neuron['rest']
    net = Network()

    neuron_type = NonSpikingNeuron(membrane_conductance=1.0, membrane_capacitance=c, resting_potential=rest)

    net.add_neuron(neuron_type)
    net.add_input(0)
    net.add_output(0)

    model = net.compile(dt)

    return model

def run_net(x_vec, model, stim):
    y_vec = np.zeros_like(x_vec)
    for i in range(len(y_vec)):
        y_vec[i] = model([stim])
    return y_vec

"""
########################################################################################################################
Main Function
"""
def graph_data_lamina(params_neuron, params_field, original_data, original_index, model_neuron, combined=True, res=5, min_angle=-10, max_angle=10, dt=0.1):
    t = np.arange(0, 1.0, dt / 1000)
    print(original_data['title'][original_index])

    # Compute shifted receptive fields and temporal response
    axis, field_2d, field_2d_shifted, field_1d, field_1d_shifted, y, y_shifted = adjust_spatiotemporal_data(t, original_data,
                                                                                                            original_index, 1.0,
                                                                                                            rest=0.0,
                                                                                                            res=res,
                                                                                                            min_angle=min_angle,
                                                                                                            max_angle=max_angle)

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["darkblue", "blue", "white", "red", "darkred"])
    norm_old = plt.Normalize(-1, 1)
    norm_new = plt.Normalize(0, 1)

    if combined:
        plt.figure()
        plt.suptitle(original_data['title'][original_index])
        grid = plt.GridSpec(5, 2)
        plt.subplot(grid[0,0])
    else:
        plt.figure()
        plt.suptitle(original_data['title'][original_index])
    plt.imshow(field_2d, extent=[axis[0], axis[-1], axis[0], axis[-1]], cmap=cmap, norm=norm_old)
    plt.colorbar()
    plt.title('Original 2d Field')
    plt.xlabel('Azimuth (deg)')
    plt.ylabel('Elevation (deg)')

    if combined:
        plt.subplot(grid[0, 1])
    else:
        plt.figure()
        plt.suptitle(original_data['title'][original_index])
    plt.imshow(field_2d_shifted, extent=[axis[0], axis[-1], axis[0], axis[-1]], cmap=cmap, norm=norm_new)
    plt.colorbar()
    plt.title('Transformed 2d Field')
    plt.xlabel('Azimuth (deg)')
    plt.ylabel('Elevation (deg)')

    if combined:
        plt.subplot(grid[1, 0])
    else:
        plt.figure()
        plt.suptitle(original_data['title'][original_index])
    plt.plot(axis, field_1d)
    plt.xlabel('Angle (deg)')
    plt.ylabel('Response')
    plt.title('Original Spatial Response')

    if combined:
        plt.subplot(grid[1, 1])
    else:
        plt.figure()
        plt.suptitle(original_data['title'][original_index])
    plt.plot(axis, field_1d_shifted)
    plt.xlabel('Angle (deg)')
    plt.ylabel('Response')
    plt.title('Scaled Spatial Response')

    if combined:
        plt.subplot(grid[2, :])
    else:
        plt.figure()
        plt.suptitle(original_data['title'][original_index])
    plt.plot(t, y, label='Response')
    plt.xlabel('t')
    plt.ylabel('Response')
    plt.title('Original Step Response')

    if combined:
        plt.subplot(grid[3, :])
    else:
        plt.figure()
        plt.suptitle(original_data['title'][original_index])
    plt.plot(t, y_shifted, label='Data')
    y_vec = run_net(t, model_neuron, original_data['polarity'][original_index])
    plt.plot(t, y_vec, label='Fit')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('Response')
    plt.title('Fitted Response')


"""
########################################################################################################################
Import Data
"""
# Original
borst = pickle.load(open('Original Data/borst_data.p', 'rb'))
borst_lamina = borst['lamina']
borst_medulla_on = borst['medullaOn']
borst_medulla_off = borst['medullaOff']

# Retina
params_neuron_retina = pickle.load(open('Retina/retina_params.p', 'rb'))

# Lamina
params_neuron_L1 = pickle.load(open('Lamina/L1_params.p', 'rb'))
params_neuron_L2 = pickle.load(open('Lamina/L2_params.p', 'rb'))
params_neuron_L3 = pickle.load(open('Lamina/L3_params.p', 'rb'))
params_neuron_L4 = pickle.load(open('Lamina/L4_params.p', 'rb'))
params_neuron_L5 = pickle.load(open('Lamina/L5_params.p', 'rb'))

params_field_L1 = pickle.load(open('Lamina/L1_field_params.p', 'rb'))
params_field_L2 = pickle.load(open('Lamina/L2_field_params.p', 'rb'))
params_field_L3 = pickle.load(open('Lamina/L3_field_params.p', 'rb'))
params_field_L4 = pickle.load(open('Lamina/L4_field_params.p', 'rb'))
params_field_L5 = pickle.load(open('Lamina/L5_field_params.p', 'rb'))

# Medulla (On)
params_neuron_Mi1 = pickle.load(open('Medulla On/Mi1_params.p', 'rb'))
params_neuron_Tm3 = pickle.load(open('Medulla On/Tm3_params.p', 'rb'))
params_neuron_Mi4 = pickle.load(open('Medulla On/Mi4_params.p', 'rb'))
params_neuron_Mi9 = pickle.load(open('Medulla On/Mi9_params.p', 'rb'))

# Medulla (Off)
params_neuron_Tm1 = pickle.load(open('Medulla Off/Tm1_params.p', 'rb'))
params_neuron_Tm2 = pickle.load(open('Medulla Off/Tm2_params.p', 'rb'))
params_neuron_Tm4 = pickle.load(open('Medulla Off/Tm4_params.p', 'rb'))
params_neuron_Tm9 = pickle.load(open('Medulla Off/Tm9_params.p', 'rb'))

"""
########################################################################################################################
Generate Models
"""
dt = 0.1

model_L1 = construct_bandpass(params_neuron_L1, borst_lamina, 0, dt)
model_L2 = construct_bandpass(params_neuron_L2, borst_lamina, 1, dt)
model_L3 = construct_lowpass(params_neuron_L3, dt)
model_L4 = construct_bandpass(params_neuron_L4, borst_lamina, 3, dt)
model_L5 = construct_bandpass(params_neuron_L5, borst_lamina, 4, dt)

"""
########################################################################################################################
Run
"""
combined = True
lamina = True
medulla_on = False
medulla_off = False

if lamina:
    graph_data_lamina(params_neuron_L1, params_field_L1, borst_lamina, 0, model_L1, combined=combined, dt=dt)
    graph_data_lamina(params_neuron_L2, params_field_L2, borst_lamina, 1, model_L2, combined=combined, dt=dt)
    graph_data_lamina(params_neuron_L3, params_field_L3, borst_lamina, 2, model_L3, combined=combined, dt=dt)
    graph_data_lamina(params_neuron_L4, params_field_L4, borst_lamina, 3, model_L4, combined=combined, dt=dt)
    graph_data_lamina(params_neuron_L5, params_field_L5, borst_lamina, 4, model_L5, combined=combined, dt=dt)

plt.show()
