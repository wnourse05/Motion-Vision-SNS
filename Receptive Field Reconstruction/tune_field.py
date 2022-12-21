import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import signal
import scipy.optimize as optimize
from sns_toolbox.networks import Network
from sns_toolbox.neurons import NonSpikingNeuron
from sns_toolbox.connections import NonSpikingSynapse

def calc_1d_point(x, A_rel, std_cen, std_sur):
    return np.exp((-x**2)/(2*std_cen**2)) - A_rel*np.exp((-x**2)/(2*std_sur**2))

def calc_2d_point(x, y, A_rel, std_cen, std_sur):
    return np.exp((-(x**2 + y**2))/(2*std_cen**2)) - A_rel*np.exp((-(x**2 + y**2))/(2*std_sur**2))

def loss_sq(actual, target):
    error = (actual-target)**2
    return error

def adjust_spatiotemporal_data(t, data, index, R, rest=0.0, res=5, min_angle=-20, max_angle=20):
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

def construct_bandpass(g_ret, del_e_ret, params_neuron, rest, dt):
    net = Network()

    neuron_ret = NonSpikingNeuron(membrane_capacitance=5.0, membrane_conductance=1.0, resting_potential=0.0)
    neuron_a = NonSpikingNeuron(membrane_capacitance=params_neuron['c_fast'], membrane_conductance=1.0, resting_potential=rest)
    neuron_b = NonSpikingNeuron(membrane_capacitance=params_neuron['c_fast'], membrane_conductance=1.0, resting_potential=rest)
    neuron_c = NonSpikingNeuron(membrane_capacitance=params_neuron['c_slow'], membrane_conductance=1.0, resting_potential=rest)
    neuron_d = NonSpikingNeuron(membrane_capacitance=params_neuron['c_fast'], membrane_conductance=1.0, resting_potential=rest)

    synapse_ret = NonSpikingSynapse(max_conductance=g_ret, reversal_potential=del_e_ret, e_lo=0.0, e_hi=1.0)
    synapse_ab = NonSpikingSynapse(max_conductance=params_neuron['g_ab'], reversal_potential=params_neuron['del_e_ab'], e_lo=0.0, e_hi=1.0)
    synapse_ac = NonSpikingSynapse(max_conductance=params_neuron['g_ac'], reversal_potential=params_neuron['del_e_ac'], e_lo=0.0, e_hi=1.0)
    synapse_bd = NonSpikingSynapse(max_conductance=params_neuron['g_bd'], reversal_potential=params_neuron['del_e_bd'], e_lo=0.0, e_hi=1.0)
    synapse_cd = NonSpikingSynapse(max_conductance=params_neuron['g_cd'], reversal_potential=params_neuron['del_e_cd'], e_lo=0.0, e_hi=1.0)

    net.add_neuron(neuron_ret)
    net.add_neuron(neuron_a)
    net.add_neuron(neuron_b, initial_value=0.0)
    net.add_neuron(neuron_c, initial_value=0.0)
    net.add_neuron(neuron_d)

    net.add_connection(synapse_ret,0,1)
    net.add_connection(synapse_ab,1,2)
    net.add_connection(synapse_ac,1,3)
    net.add_connection(synapse_bd,2,4)
    net.add_connection(synapse_cd,3,4)

    net.add_input(0)
    net.add_output(4)

    model = net.compile(dt)

    return model

def construct_lowpass(g_ret, del_e_ret, params_neuron, dt):
    net = Network()

    neuron_ret = NonSpikingNeuron(membrane_capacitance=5.0, membrane_conductance=1.0, resting_potential=0.0)
    neuron_type = NonSpikingNeuron(membrane_conductance=1.0, membrane_capacitance=params_neuron['c'], resting_potential=params_neuron['rest'])

    synapse = NonSpikingSynapse(max_conductance=g_ret, reversal_potential=del_e_ret, e_lo=0.0, e_hi=1.0)

    net.add_neuron(neuron_ret)
    net.add_neuron(neuron_type)

    net.add_connection(synapse,0,1)

    net.add_input(0)
    net.add_output(1)

    model = net.compile(dt)

    return model

def run_net(x_vec, model, stim):
    y_vec = np.zeros_like(x_vec)
    model.reset()
    for i in range(len(y_vec)):
        y_vec[i] = model([stim])
    return y_vec

def run_bandpass(x_vec, g_ret, del_e_ret, params_neuron, rest, stim):
    dt = (x_vec[1] - x_vec[0]) * 1000  # convert from seconds to ms
    model = construct_bandpass(g_ret, del_e_ret, params_neuron, rest, dt)
    y_vec = run_net(x_vec, model, stim)
    return y_vec

def peak_bandpass(x_vec, g_ret, del_e_ret, params_neuron, rest, stim, polarity):
    y_vec = run_bandpass(x_vec, g_ret, del_e_ret, params_neuron, rest, stim)
    peak = polarity * np.max(polarity * y_vec)
    # print(peak)
    return peak
def peak_error_bandpass(x_vec, g_ret, del_e_ret, params_neuron, rest, stim, polarity, target, loss_fn):
    peak = peak_bandpass(x_vec, g_ret, del_e_ret, params_neuron, rest, stim, polarity)
    error = loss_fn(peak, target)
    return  error

def tune_bandpass(target, t, params_neuron, data_ref, index, plot, debug):
    polarity = data_ref['polarity'][index]
    rest = (-0.5 * polarity + 0.5)
    stim = 1.0
    bound_g_lo = 0.0
    bound_g_hi = 1.0
    bound_del_e_lo = rest - 3
    bound_del_e_hi = rest + 3
    slope_g = bound_g_hi - bound_g_lo
    slope_del_e = bound_del_e_hi - bound_del_e_lo

    f = lambda x: peak_error_bandpass(t, slope_g * x[0] + bound_g_lo, slope_del_e * x[1] + bound_del_e_lo, params_neuron,
                                      rest, stim, polarity, target, loss_sq)

    bounds = ((0.0, 1.0), (0.0, 1.0))
    x0 = np.array([0.5, 1.0])

    # Ok minimizers: 'L-BFGS-B', 'Nelder-Mead', 'TNC', 'Powell', 'trust-constr'
    result = optimize.minimize(f, x0, method='Powell', bounds=bounds, options={'disp': debug})

    g_ret = slope_g * result.x[0] + bound_g_lo
    del_e_ret = slope_del_e * result.x[1] + bound_del_e_lo

    peak = peak_bandpass(t, g_ret, del_e_ret, params_neuron, rest, stim, polarity)

    error_sq = peak_error_bandpass(t, g_ret, del_e_ret, params_neuron, rest, stim, polarity, target, loss_sq)

    # if plot:
    #     plt.figure()
    #     plt.plot(t, run_bandpass(t, g_ret, del_e_ret, params_L1, rest, stim))

    return g_ret, del_e_ret, peak, error_sq

def run_lowpass(x_vec, g_ret, del_e_ret, params_neuron, stim):
    dt = (x_vec[1] - x_vec[0]) * 1000  # convert from seconds to ms
    model = construct_lowpass(g_ret, del_e_ret, params_neuron,dt)
    y_vec = run_net(x_vec, model, stim)
    return y_vec

def peak_lowpass(x_vec, g_ret, del_e_ret, params_neuron, stim):
    y_vec = run_lowpass(x_vec, g_ret, del_e_ret, params_neuron, stim)
    peak = y_vec[-1]
    return peak

def peak_error_lowpass(x_vec, g_ret, del_e_ret, params_neuron, stim, target, loss_fn):
    peak = peak_lowpass(x_vec, g_ret, del_e_ret, params_neuron, stim)
    error = loss_fn(peak, target)
    return error

def tune_lowpass(target, t, params_neuron, data_ref, index, plot, debug):
    polarity = data_ref['polarity'][index]
    rest = (-0.5 * polarity + 0.5)
    stim = 1.0
    bound_g_lo = 0.0
    bound_g_hi = 1.0
    bound_del_e_lo = rest - 3
    bound_del_e_hi = rest + 3
    slope_g = bound_g_hi - bound_g_lo
    slope_del_e = bound_del_e_hi - bound_del_e_lo

    f = lambda x: peak_error_lowpass(t, slope_g * x[0] + bound_g_lo, slope_del_e * x[1] + bound_del_e_lo,
                                      params_neuron,
                                      stim, target, loss_sq)

    bounds = ((0.0, 1.0), (0.0, 1.0))
    x0 = np.array([0.5, 0.0])

    # Ok minimizers: 'L-BFGS-B', 'Nelder-Mead', 'TNC', 'Powell', 'trust-constr'
    result = optimize.minimize(f, x0, method='Powell', bounds=bounds, options={'disp': debug})

    g_ret = slope_g * result.x[0] + bound_g_lo
    del_e_ret = slope_del_e * result.x[1] + bound_del_e_lo

    peak = peak_lowpass(t, g_ret, del_e_ret, params_neuron, stim)

    error_sq = peak_error_lowpass(t, g_ret, del_e_ret, params_neuron, stim, target, loss_sq)

    # if plot:
    #     plt.figure()
    #     plt.plot(t, run_bandpass(t, g_ret, del_e_ret, params_L1, rest, stim))

    return g_ret, del_e_ret, peak, error_sq

def convert_corner_to_full(matrix):
    num_rows, _ = matrix.shape
    full_matrix = np.zeros([2*num_rows-1, 2*num_rows-1])
    full_matrix[num_rows-1:, num_rows-1:] = matrix
    full_matrix[num_rows-1:,:num_rows] = np.flip(matrix,1)
    full_matrix[:num_rows, :num_rows] = np.flip(matrix)
    full_matrix[:num_rows, num_rows-1:] = np.flip(matrix,0)

    return full_matrix

def tune_neuron(layer_data, index, params_neuron, plot, debug):
    dt = 0.1
    res = 5
    min_angle = 0
    max_angle = 10

    # rng = np.random.default_rng(seed=0)
    t = np.arange(0,1.0,dt/1000)
    print(layer_data['title'][index])

    # Compute shifted receptive fields and temporal response
    print('Field')
    axis, field_2d, field_2d_shifted, field_1d, field_1d_shifted, y, y_shifted = adjust_spatiotemporal_data(t, layer_data, index, 1.0, rest=0.0, res=res, min_angle=min_angle, max_angle=max_angle)

    num_rows, num_cols = field_2d_shifted.shape
    g_ret = np.zeros_like(field_2d_shifted)
    del_e_ret = np.zeros_like(field_2d_shifted)
    peak = np.zeros_like(field_2d_shifted)
    error_sq = np.zeros_like(field_2d_shifted)
    for i in range(num_rows):
        for j in range(num_cols):
            target = field_2d_shifted[i,j]
            if debug:
                print('\nRow %i, Column %i' % (i,j))
                print('Target: %f' % target)
                print('Started minimization')
            if layer_data['tau_hp'][index] is None:
                g_ret[i,j], del_e_ret[i,j], peak[i,j], error_sq[i,j] = tune_lowpass(target, t, params_neuron, layer_data, index, plot, debug)
            else:
                g_ret[i,j], del_e_ret[i,j], peak[i,j], error_sq[i,j] = tune_bandpass(target, t, params_neuron, layer_data, index, plot, debug)
            if debug:
                print('Minimization finished')

    if debug:
        print(g_ret)
        print(del_e_ret)
        print(peak)
        print(error_sq)

    g_ret_full = convert_corner_to_full(g_ret)
    del_e_ret_full = convert_corner_to_full(del_e_ret)

    if debug:
        print(g_ret_full)
        print(del_e_ret_full)

    output = {'g': g_ret_full, 'del_e': del_e_ret_full}
    filename = layer_data['title'][index] + '_field_params.p'
    pickle.dump(output, open(filename, 'wb'))

    if plot:
        plt.figure()
        grid = plt.GridSpec(3, 2)
        plt.suptitle(layer_data['title'][index])

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["darkblue", "blue", "white", "red", "darkred"])
        norm_old = plt.Normalize(-1, 1)
        norm_new = plt.Normalize(0, 1)

        plt.subplot(grid[0, 0])
        plt.imshow(field_2d, extent=[axis[0], axis[-1], axis[-1], axis[0]], cmap=cmap, norm=norm_old, interpolation='none')
        plt.colorbar()
        plt.title('Original 2d Field')
        plt.xlabel('Azimuth (deg)')
        plt.ylabel('Elevation (deg)')

        plt.subplot(grid[0, 1])
        plt.imshow(field_2d_shifted, extent=[axis[0], axis[-1], axis[-1], axis[0]], cmap=cmap, norm=norm_new, interpolation='none')
        plt.colorbar()
        plt.title('Transformed 2d Field')
        plt.xlabel('Azimuth (deg)')
        plt.ylabel('Elevation (deg)')

        plt.subplot(grid[1, 0])
        plt.plot(axis, field_1d)
        plt.xlabel('Angle (deg)')
        plt.ylabel('Response')
        plt.title('Original Spatial Response')

        plt.subplot(grid[1, 1])
        plt.plot(axis, field_1d_shifted)
        plt.xlabel('Angle (deg)')
        plt.ylabel('Response')
        plt.title('Scaled Spatial Response')

        plt.subplot(grid[2, 0])
        plt.imshow(peak, extent=[axis[0], axis[-1], axis[-1], axis[0]], cmap=cmap, norm=norm_new,
                   interpolation='none')
        plt.colorbar()
        plt.title('Optimized 2d Field')
        plt.xlabel('Azimuth (deg)')
        plt.ylabel('Elevation (deg)')

        plt.subplot(grid[2, 1])
        plt.imshow(error_sq, extent=[axis[0], axis[-1], axis[-1], axis[0]], interpolation='none')
        plt.colorbar()
        plt.title('Squared Error')
        plt.xlabel('Azimuth (deg)')
        plt.ylabel('Elevation (deg)')

borst_data = pickle.load(open('borst_data.p', 'rb'))
lamina_data = borst_data['lamina']
params_L1 = pickle.load(open('L1_params.p', 'rb'))
params_L2 = pickle.load(open('L2_params.p', 'rb'))
params_L3 = pickle.load(open('L3_params.p', 'rb'))
params_L4 = pickle.load(open('L4_params.p', 'rb'))
params_L5 = pickle.load(open('L5_params.p', 'rb'))

plot = True
debug = True

tune_neuron(lamina_data, 0, params_L1, plot, debug)
tune_neuron(lamina_data, 1, params_L2, plot, debug)
tune_neuron(lamina_data, 2, params_L3, plot, debug)
tune_neuron(lamina_data, 3, params_L4, plot, debug)
# tune_neuron(lamina_data, 4, params_L5, plot, debug)

plt.show()
