import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import signal
import sys
import scipy.optimize as optimize
from sns_toolbox.networks import Network
from sns_toolbox.neurons import NonSpikingNeuron
from sns_toolbox.connections import NonSpikingSynapse

# Personal stuff to send an email once data collection is finished
sys.path.extend(['/home/will'])
from email_utils import send_email

def calc_1d_point(x, A_rel, std_cen, std_sur):
    return np.exp((-x**2)/(2*std_cen**2)) - A_rel*np.exp((-x**2)/(2*std_sur**2))

def calc_2d_point(x, y, A_rel, std_cen, std_sur):
    return np.exp((-(x**2 + y**2))/(2*std_cen**2)) - A_rel*np.exp((-(x**2 + y**2))/(2*std_sur**2))

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

def construct_bandpass(g_ab, g_ac, g_bd, g_cd, del_e_ab, del_e_ac, del_e_bd, del_e_cd, c_a, c_b, c_c, c_d, rest, dt):
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

def construct_lowpass(c, rest, dt):
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

def test_lowpass(x_vec, c, rest, stim):
    dt = (x_vec[1] - x_vec[0])*1000 # convert from seconds to ms
    model = construct_lowpass(c, rest, dt)
    y_vec = run_net(x_vec, model, stim)
    return y_vec

def test_bandpass(x_vec, g_ab, g_ac, g_bd, g_cd, del_e_ab, del_e_ac, del_e_bd, del_e_cd, c_slow, c_fast, stim, rest):#, c_a, c_b, c_c, c_d):
    dt = (x_vec[1] - x_vec[0]) * 1000  # convert from seconds to ms
    c_a = c_fast
    c_b = c_a
    c_c = c_slow
    c_d = c_a

    model = construct_bandpass(g_ab, g_ac, g_bd, g_cd, del_e_ab, del_e_ac, del_e_bd, del_e_cd, c_a, c_b, c_c, c_d, rest, dt)
    y_vec = run_net(x_vec, model, stim)
    return y_vec

def tune_neuron(data, index, res=5, min_angle=-20, max_angle=20, plot=True, dt=0.1):
    rng = np.random.default_rng(seed=0)
    t = np.arange(0,1.0,dt/1000)
    print(data['title'][index])

    # Compute shifted receptive fields and temporal response
    print('Field')
    axis, field_2d, field_2d_shifted, field_1d, field_1d_shifted, y, y_shifted = adjust_spatiotemporal_data(t, data, index, 1.0, rest=0.0, res=res, min_angle=min_angle, max_angle=max_angle)

    # Tune the neural parameters
    print('Neuron')
    if data['tau_hp'][index] is None:
        # model = construct_lowpass(5, 0, t[1]-t[0])
        # out = run_net(t, model, R)
        bounds = ([0.0, 0.0], [np.inf, 1.0])
        f = lambda x_vec, c, rest: test_lowpass(x_vec, c, rest, data['polarity'][index])
    else:
        f = lambda x_vec, g_bd, g_cd, del_e_bd, del_e_cd: test_bandpass(x_vec, 1.0, 1.0, g_bd, g_cd, data['polarity'][index], data['polarity'][index], del_e_bd, del_e_cd, data['tau_hp'][index]*1000, data['tau_lp'][index]*1000, data['polarity'][index], (-0.5*data['polarity'][index]+0.5))
        # initial_guess = [50.0, 300.0]
        bound_g_lo = 0.0
        bound_g_hi = 1.0
        bound_del_e_in_lo = -4.0
        bound_del_e_in_hi = 0.0
        bound_del_e_ex_lo = 1.0
        bound_del_e_ex_hi = 5.0
        bound_c_lo = dt
        bound_c_hi = np.inf
        if data['polarity'][index] > 0:
            #           g_ab        g_cd          del_e_bd            del_e_cd
            bounds = ([bound_g_lo, bound_g_lo, bound_g_lo, bound_del_e_ex_lo, bound_del_e_ex_lo, bound_del_e_in_lo],#, bound_c_lo, bound_c_lo],# bound_c_lo, bound_c_lo],
                      [bound_g_hi, bound_g_hi, bound_g_hi, bound_del_e_ex_hi, bound_del_e_ex_hi, bound_del_e_in_hi])#, bound_c_hi, bound_c_hi])#, bound_c_hi, bound_c_hi])
            f = lambda x_vec, g_a, g_bd, g_cd, del_e_a, del_e_bd, del_e_cd: test_bandpass(x_vec, g_a, g_a, g_bd, g_cd,
                                                                            del_e_a,
                                                                            del_e_a, del_e_bd, del_e_cd,
                                                                            data['tau_hp'][index] * 1000,
                                                                            data['tau_lp'][index] * 1000,
                                                                            data['polarity'][index],
                                                                            (-0.5 * data['polarity'][index] + 0.5))
        else:
            bounds = ([bound_g_lo, bound_g_lo, bound_del_e_in_lo, bound_del_e_ex_lo],# bound_c_lo, bound_c_lo],#, bound_c_lo, bound_c_lo],
                      [bound_g_hi, bound_g_hi, bound_del_e_in_hi, bound_del_e_ex_hi])#, bound_c_hi, bound_c_hi])#, bound_c_hi, bound_c_hi])
            f = lambda x_vec, g_bd, g_cd, del_e_bd, del_e_cd: test_bandpass(x_vec, 1.0, 1.0, g_bd, g_cd,
                                                                            data['polarity'][index],
                                                                            data['polarity'][index], del_e_bd, del_e_cd,
                                                                            data['tau_hp'][index] * 1000,
                                                                            data['tau_lp'][index] * 1000,
                                                                            data['polarity'][index],
                                                                            (-0.5 * data['polarity'][index] + 0.5))

    params_neuron, cov = optimize.curve_fit(f, t, y_shifted, bounds=bounds)
    print(params_neuron)

    # Save results
    if data['tau_hp'][index] is None:
        output = {'c': params_neuron[0],
                  'rest': params_neuron[1]}
    else:
        if data['polarity'][index] > 0:
            output = {'g_ab': params_neuron[0],
                      'g_ac': params_neuron[0],
                      'g_bd': params_neuron[1],
                      'g_cd': params_neuron[2],
                      'del_e_ab': params_neuron[3],
                      'del_e_ac': params_neuron[3],
                      'del_e_bd': params_neuron[4],
                      'del_e_cd': params_neuron[5],
                      'c_fast': data['tau_lp'][index] * 1000,
                      'c_slow': data['tau_hp'][index] * 1000}
        else:
            output = {'g_ab': 1.0,
                      'g_ac': 1.0,
                      'g_bd': params_neuron[0],
                      'g_cd': params_neuron[1],
                      'del_e_ab': -1.0,
                      'del_e_ac': -1.0,
                      'del_e_bd': params_neuron[2],
                      'del_e_cd': params_neuron[3],
                      'c_fast': data['tau_lp'][index]*1000,
                      'c_slow': data['tau_hp'][index]*1000}

    filename = data['title'][index] + '_params.p'

    pickle.dump(output, open(filename, 'wb'))

    if plot:
        plt.figure()
        grid = plt.GridSpec(5,2)
        plt.suptitle(data['title'][index])

        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["darkblue", "blue", "white", "red", "darkred"])
        norm_old = plt.Normalize(-1, 1)
        norm_new = plt.Normalize(0,1)

        plt.subplot(grid[0,0])
        plt.imshow(field_2d, extent=[axis[0], axis[-1], axis[0], axis[-1]], cmap=cmap, norm=norm_old)
        plt.colorbar()
        plt.title('Original 2d Field')
        plt.xlabel('Azimuth (deg)')
        plt.ylabel('Elevation (deg)')

        plt.subplot(grid[0, 1])
        plt.imshow(field_2d_shifted, extent=[axis[0], axis[-1], axis[0], axis[-1]], cmap=cmap, norm=norm_new)
        plt.colorbar()
        plt.title('Transformed 2d Field')
        plt.xlabel('Azimuth (deg)')
        plt.ylabel('Elevation (deg)')

        plt.subplot(grid[1,0])
        plt.plot(axis, field_1d)
        plt.xlabel('Angle (deg)')
        plt.ylabel('Response')
        plt.title('Original Spatial Response')

        plt.subplot(grid[1,1])
        plt.plot(axis, field_1d_shifted)
        plt.xlabel('Angle (deg)')
        plt.ylabel('Response')
        plt.title('Scaled Spatial Response')

        plt.subplot(grid[2,0])
        plt.plot(t, y, label='Response')
        plt.xlabel('t')
        plt.ylabel('Response')
        plt.title('Original Step Response')

        plt.subplot(grid[2,1])
        plt.plot(t, y_shifted)
        plt.xlabel('t')
        plt.ylabel('Response')
        plt.title('Scaled Step Response')

        plt.subplot(grid[3,:])
        plt.plot(t, y_shifted, label='Data')
        plt.plot(t, f(t, *params_neuron), label='Fit')
        plt.legend()
        plt.xlabel('t')
        plt.ylabel('Response')
        plt.title('Fitted Response')

borst_data = pickle.load(open('borst_data.p', 'rb'))
medulla_on = borst_data['medullaOn']

res = 5

# tune_neuron(medulla_on, 0, min_angle=-(res*2), max_angle=res*2, res=res)
try:
    tune_neuron(medulla_on, 1, min_angle=-(res*2), max_angle=res*2, res=res)
except:
    print('Tm3 Failed. Moving On')
try:
    tune_neuron(medulla_on, 2, min_angle=-(res*2), max_angle=res*2, res=res)
except:
    print('Mi4 Failed. Moving On')
try:
    tune_neuron(medulla_on, 3, min_angle=-(res*2), max_angle=res*2, res=res)
except:
    print('Mi9 Failed. Moving On')

send_email('wrn13@case.edu')
plt.show()
