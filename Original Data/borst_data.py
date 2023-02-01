"""
Recreate the spatial receptive fields found in
"The Temporal Tuning of the Drosophila Motion Detectors Is Determined by the Dynamics of Their Input Elements" and
"Dynamic Signal Compression for Robust Motion Vision in Flies"
William Nourse
April 29th, 2022
"""
import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import signal

"""Helper Functions"""

def FWHM_to_std(FWHM):
    std = FWHM/(2*np.sqrt(2*np.log(2)))
    # print(std)
    return std

def calc_2d_point(x, y, A_rel, std_cen, std_sur):
    return np.exp((-(x**2 + y**2))/(2*std_cen**2)) - A_rel*np.exp((-(x**2 + y**2))/(2*std_sur**2))

def calc_1d_point(x, A_rel, std_cen, std_sur):
    return np.exp((-x**2)/(2*std_cen**2)) - A_rel*np.exp((-x**2)/(2*std_sur**2))

def plot_fields(data, res, max_angle, min_angle, cmap, norm, num_neurons):
    # fig = plt.figure()
    # num_neurons = len(data['title'])
    axis = np.arange(min_angle, max_angle + res, res)
    for neuron in range(num_neurons):
        plt.subplot(5, num_neurons, neuron+1)
        field_2d = np.zeros([len(axis), len(axis)])
        for i in range(len(axis)):
            for j in range(len(axis)):
                field_2d[i, j] = data['polarity'][neuron] * calc_2d_point(axis[i], axis[j], data['A_rel'][neuron], data['std_cen'][neuron], data['std_sur'][neuron])
        plt.imshow(field_2d, extent=[axis[0], axis[-1], axis[0], axis[-1]], cmap=cmap, norm=norm)
        plt.colorbar()
        plt.title(data['title'][neuron])
        plt.xlabel('Azimuth (deg)')
        plt.ylabel('Elevation (deg)')

        plt.subplot(5, num_neurons, neuron+1+num_neurons)
        field_1d = np.zeros(len(axis))
        for i in range(len(axis)):
            field_1d[i] = data['polarity'][neuron] * calc_1d_point(axis[i], data['A_rel'][neuron], data['std_cen'][neuron], data['std_sur'][neuron])

        plt.plot(axis, field_1d)
        plt.xlabel('Angle (deg)')
        plt.ylabel('Response')
    # return fig

def plot_step_response(data,  num_neurons):
    # nsamples = int(T * fs)
    # t = np.linspace(0, T, nsamples, endpoint=False)

    for neuron in range(num_neurons):
        lowcut = 1/(data['tau_lp'][neuron])
        if data['tau_hp'][neuron] is None:
            # print(lowcut)
            b,a = signal.butter(1,lowcut,'low',analog=True)
        else:
            highcut = 1 / (data['tau_hp'][neuron])
            # print(lowcut, highcut)
            b,a = signal.butter(1, [highcut, lowcut], 'band', analog=True)
        t, y = signal.step((b,a))
        x = np.ones_like(t)
        plt.subplot(5,num_neurons, neuron+1+2*num_neurons)
        plt.plot(t, y*data['polarity'][neuron], label='Response')
        plt.plot(t, x*data['polarity'][neuron], color='black', linestyle='--', label='Step')

        plt.legend()
        plt.xlabel('t')
        plt.ylabel('Response')

def plot_freq_response(data, num_neurons):
    for neuron in range(num_neurons):
        lowcut = 1/(data['tau_lp'][neuron])
        if data['tau_hp'][neuron] is None:
            # print(lowcut)
            b,a = signal.butter(1,lowcut,'low',analog=True)
        else:
            highcut = 1 / (data['tau_hp'][neuron])
            # print(lowcut, highcut)
            b,a = signal.butter(1, [highcut, lowcut], 'band', analog=True)
        w,h = signal.freqs(b,a, worN=np.linspace(0.1*2*np.pi, 10*2*np.pi, num=2000))
        plt.subplot(5,num_neurons,neuron+1+3*num_neurons)
        plt.semilogx(w,20*np.log10(abs(h)))
        plt.axhline(-3,color='black',ls='--')
        plt.ylim([-30,5])
        plt.xlabel('Frequency (rad/s)')
        plt.ylabel('Amplitude Response')

def plot_field_ratio(data, res, max_angle, min_angle, num_neurons):
    axis = np.arange(0, max((abs(min_angle), abs(max_angle))) + res, res)
    for neuron in range(num_neurons):
        plt.subplot(5, num_neurons, neuron + 1 + 4*num_neurons)

        field_1d = np.zeros(len(axis))
        for i in range(len(axis)):
            field_1d[i] = calc_1d_point(axis[i], data['A_rel'][neuron],data['std_cen'][neuron], data['std_sur'][neuron])

        peak_mag = max(abs(field_1d))
        field_norm = abs(field_1d)/peak_mag

        plt.plot(axis, field_norm)
        plt.xlabel('Angle (deg)')
        plt.ylabel('Response Ratio')

def plot_properties(data, res, max_angle, min_angle, cmap, norm):
    fig = plt.figure()
    num_neurons = len(data['title'])
    # t_max = 1
    # fs = 1/dt
    # nsamples = int(t_max/dt)
    # t = np.linspace(0, t_max, num=nsamples)
    plot_fields(data, res, max_angle, min_angle, cmap, norm, num_neurons)
    plot_step_response(data, num_neurons)
    plot_freq_response(data, num_neurons)
    plot_field_ratio(data, res, max_angle, min_angle, num_neurons)
    return fig

# def plot_freq_response(data, dt):


"""Data"""

# Model data from supplemental material
lamina = {'title':      ['L1', 'L2', 'L3', 'L4', 'L5'],
          'A_rel':      [0.012, 0.013, 0.193, 0.046, 0.035],
          'std_cen':    [2.679, 2.861, 2.514, 3.633, 2.883],
          'std_sur':    [17.473, 12.449, 6.423, 13.911, 13.325],
          'polarity':   [-1, -1, -1, -1, 1],
          'tau_hp':     [0.391, 0.288, None,  0.381, 0.127],
          'tau_lp':     [0.038, 0.058, 0.054, 0.023, 0.042]}

title =          ['Mi1','Tm3','Mi4', 'Mi9','Tm1', 'Tm2', 'Tm4', 'Tm9']
A_rel = np.array([0.022, 0.0, 0.132, 0.063, 0.04, 0.035, 0.054, 0.046])
FWHM_cen = np.array([6.81, 11.91, 6.47, 6.37, 8.12, 7.93, 11.45, 6.92])
FWHM_sur = np.array([28.81, 1.0, 16.14, 23.98, 27.14, 30.52, 34.62, 23.78])
polarity = np.array([1, 1, 1, -1, -1, -1, -1, -1])
# std_cen = FWHM_cen/(2*np.sqrt(2*np.log(2)))
# std_sur = FWHM_sur/(2*np.sqrt(2*np.log(2)))

medulla_on = {'title':      title[:4],
              'A_rel':      A_rel[:4],
              'std_cen':    FWHM_to_std(FWHM_cen[:4]),
              'std_sur':    FWHM_to_std(FWHM_sur[:4]),
              'polarity':   polarity[:4],
              'tau_hp':     [0.318, 0.260, None,  None],
              'tau_lp':     [0.054, 0.027, 0.038, 0.077]}

medulla_off = {'title':      title[4:],
               'A_rel':      A_rel[4:],
               'std_cen':    FWHM_to_std(FWHM_cen[4:]),
               'std_sur':    FWHM_to_std(FWHM_sur[4:]),
               'polarity':   polarity[4:],
               'tau_hp':     [0.296, 0.153, 0.249,  None],
               'tau_lp':     [0.044, 0.014, 0.024, 0.017]}

data = {'lamina': lamina,
        'medullaOn': medulla_on,
        'medullaOff': medulla_off}

pickle.dump(data, open('borst_data.p', 'wb'))

"""Plotting"""

res = 5
max_angle = 20
min_angle = -20
dt = 0.01/1000
# fs = 1/dt
# axis = np.arange(min_angle, max_angle+res, res)

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("",["darkblue","blue","white","red","darkred"])
norm = plt.Normalize(-1,1)

fig_lamina = plot_properties(lamina, res, max_angle, min_angle, cmap, norm)
fig_medulla_on = plot_properties(medulla_on, res, max_angle, min_angle, cmap, norm)
fig_medulla_off = plot_properties(medulla_off, res, max_angle, min_angle, cmap, norm)

plt.show()