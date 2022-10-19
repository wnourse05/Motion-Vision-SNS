from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.optimize import curve_fit

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def forward(x, V_last, input_connectivity, g_max_non, e_lo, e_hi, del_e, time_factor_membrane, g_m, V_rest, i_b, output_voltage_connectivity):
    i_app = np.matmul(input_connectivity, x)  # Apply external current sources to their destinations
    g_syn = np.maximum(0, np.minimum(g_max_non * ((V_last - e_lo) / (e_hi - e_lo)), g_max_non))

    i_syn = np.sum(g_syn * del_e, axis=1) - V_last * np.sum(g_syn, axis=1)

    V = V_last + time_factor_membrane * (-g_m * (V_last - V_rest) + i_b + i_syn + i_app)  # Update membrane potential

    outputs = np.matmul(output_voltage_connectivity, V)

    return outputs, V

def run_net(x_vec, V_init, input_connectivity, g_max_non, e_lo, e_hi, del_e, time_factor_membrane, g_m, V_rest, i_b, output_voltage_connectivity):
    data = np.zeros_like(x_vec)
    V_last = V_init
    for i in range(len(data)):
        data[i], V_last = forward([x_vec[i]], V_last, input_connectivity, g_max_non, e_lo, e_hi, del_e, time_factor_membrane, g_m, V_rest, i_b, output_voltage_connectivity)
    return data

def build_net(g_a, g_b, del_e_a, del_e_b, c_fast, c_slow, dt):
    num_neurons = 4
    V_init = np.zeros(num_neurons)
    input_connectivity = np.zeros([num_neurons,1])
    input_connectivity[0,0] = 1.0

    g_max_non = np.zeros([num_neurons, num_neurons])
    g_max_non[1,0] = g_a
    g_max_non[2,0] = g_a
    g_max_non[3,1] = g_a
    g_max_non[3,2] = g_b

    e_lo = np.zeros_like(V_init)
    e_hi = np.ones_like(V_init)

    del_e = np.zeros_like(g_max_non)
    del_e[1, 0] = del_e_a
    del_e[2, 0] = del_e_a
    del_e[3, 1] = del_e_a
    del_e[3, 2] = del_e_b

    c_m = np.array([c_fast, c_fast, c_slow, c_fast])
    time_factor_membrane = dt/c_m

    g_m = np.ones_like(V_init)
    V_rest = np.zeros_like(V_init)
    i_b = np.zeros_like(V_init)

    output_voltage_connectivity = np.zeros([1, num_neurons])
    output_voltage_connectivity[0,3] = 1

    return V_init, input_connectivity, g_max_non, e_lo, e_hi, del_e, time_factor_membrane, g_m, V_rest, i_b, output_voltage_connectivity

def test_net(x_vec, g_a, g_b, del_e_a, del_e_b, c_fast, c_slow):
    dt = 0.01
    V_init, input_connectivity, g_max_non, e_lo, e_hi, del_e, time_factor_membrane, g_m, V_rest, i_b, output_voltage_connectivity = build_net(g_a, g_b, del_e_a, del_e_b, c_fast, c_slow, dt)
    data = run_net(x_vec, V_init, input_connectivity, g_max_non, e_lo, e_hi, del_e, time_factor_membrane, g_m, V_rest, i_b, output_voltage_connectivity)
    return data

def get_ground_truth(T,fs,lowcut,highcut):
    nsamples = int(T * fs)
    t = np.linspace(0, T, nsamples, endpoint=False)
    x = np.ones_like(t)

    y = butter_bandpass_filter(x, lowcut, highcut, fs, order=1)

    return t,x,y

# Sample rate and desired cutoff frequencies (in Hz).
fs = 5000.0
lowcut = 50.0
highcut = 500.0
T = 0.03

t,x,y = get_ground_truth(T, fs, lowcut, highcut)

params, cov = curve_fit(test_net, x, y, bounds=([0., 0., 0., -3., 0., 0.],[1., 1., 3., 0., np.inf, np.inf]))

print(params)
# g_a, g_b, del_e_a, del_e_b, c_fast, c_slow
g_a = params[0]
g_b = params[1]
del_e_a = params[2]
del_e_b = params[3]
c_fast = params[4]
c_slow = params[5]

result = {'g_a': g_a,
          'g_b': g_b,
          'del_e_a': del_e_a,
          'del_e_b': del_e_b,
          'c_fast': c_fast,
          'c_slow': c_slow}

pickle.dump(result, open("bandpass_params.p", 'wb'))

data = test_net(x, g_a, g_b, del_e_a, del_e_b, c_fast, c_slow)

plt.figure()
plt.plot(t,y, label='Ideal')
plt.plot(t,x,color='black',linestyle='--', label='Step')
plt.plot(t, data, label='Neural Fit')
plt.legend()
plt.show()

# old stuff
# # Plot the frequency response for a few different orders.
# plt.figure(1)
# plt.clf()
# for order in [1]:
#     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#     w, h = freqz(b, a, worN=2000)
#     plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
#
# plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
#          '--', label='sqrt(0.5)')
# plt.xlim([0,1000])
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Gain')
# plt.grid(True)
# plt.legend(loc='best')

# Filter a noisy signal.
# T = 0.03
# nsamples = int(T * fs)
# t = np.linspace(0, T, nsamples, endpoint=False)
# a = 0.02
# f0 = 600.0
# x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
# x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
# x += a * np.cos(2 * np.pi * f0 * t + .11)
# x += 0.03 * np.cos(2 * np.pi * 2000 * t)
# x = np.ones_like(t)
# plt.figure(2)
# plt.clf()
# plt.plot(t, x, label='Noisy signal')
#
# y = butter_bandpass_filter(x, lowcut, highcut, fs, order=1)
# plt.plot(t, y, label='Filtered signal')
# plt.xlabel('time (seconds)')
# # plt.hlines([-a, a], 0, T, linestyles='--')
# # plt.grid(True)
# # plt.axis('tight')
# plt.ylim([0,1])
# plt.legend(loc='upper left')

# plt.show()