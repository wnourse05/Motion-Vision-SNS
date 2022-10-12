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

"""Helper Functions"""

def FWHM_to_std(FWHM):
    std = FWHM/(2*np.sqrt(2*np.log(2)))
    # print(std)
    return std

def calc_2d_point(x, y, A_rel, std_cen, std_sur):
    return np.exp((-(x**2 + y**2))/(2*std_cen**2)) - A_rel*np.exp((-(x**2 + y**2))/(2*std_sur**2))

def calc_1d_point(x, A_rel, std_cen, std_sur):
    return np.exp((-x**2)/(2*std_cen**2)) - A_rel*np.exp((-x**2)/(2*std_sur**2))

def plot_fields(data, res, max_angle, min_angle, cmap, norm):
    fig = plt.figure()
    num_neurons = len(data['title'])
    axis = np.arange(min_angle, max_angle + res, res)
    for neuron in range(num_neurons):
        plt.subplot(2, num_neurons, neuron+1)
        field_2d = np.zeros([len(axis), len(axis)])
        for i in range(len(axis)):
            for j in range(len(axis)):
                field_2d[i, j] = data['polarity'][neuron] * calc_2d_point(axis[i], axis[j], data['A_rel'][neuron], data['std_cen'][neuron], data['std_sur'][neuron])
        plt.imshow(field_2d, extent=[axis[0], axis[-1], axis[0], axis[-1]], cmap=cmap, norm=norm)
        plt.colorbar()
        plt.title(data['title'][neuron])
        plt.xlabel('Azimuth (deg)')
        plt.ylabel('Elevation (deg)')

        plt.subplot(2, num_neurons, neuron+1+num_neurons)
        field_1d = np.zeros(len(axis))
        for i in range(len(axis)):
            field_1d[i] = data['polarity'][neuron] * calc_1d_point(axis[i], data['A_rel'][neuron], data['std_cen'][neuron], data['std_sur'][neuron])

        plt.plot(axis, field_1d)
        plt.xlabel('Angle (deg)')
        plt.ylabel('Response')
    return fig


"""Data"""

# Model data from supplemental material
lamina = {'title':      ['L1', 'L2', 'L3', 'L4', 'L5'],
          'A_rel':      [0.012, 0.013, 0.193, 0.046, 0.035],
          'std_cen':    [2.679, 2.861, 2.514, 3.633, 2.883],
          'std_sur':    [17.473, 12.449, 6.423, 13.911, 13.325],
          'polarity':   [-1, -1, -1, -1, 1]}

title = ['Mi1', 'Tm3', 'Mi4', 'Mi9', 'Tm1', 'Tm2', 'Tm4', 'Tm9']
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
              'polarity':   polarity[:4]}

medulla_off = {'title':      title[4:],
               'A_rel':      A_rel[4:],
               'std_cen':    FWHM_to_std(FWHM_cen[4:]),
               'std_sur':    FWHM_to_std(FWHM_sur[4:]),
               'polarity':   polarity[4:]}

"""Plotting"""

res = 5
max_angle = 20
min_angle = -20
# axis = np.arange(min_angle, max_angle+res, res)

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("",["darkblue","blue","white","red","darkred"])
norm = plt.Normalize(-1,1)

fig_lamina = plot_fields(lamina, res, max_angle, min_angle, cmap, norm)
fig_medulla_on = plot_fields(medulla_on, res, max_angle, min_angle, cmap, norm)
fig_medulla_off = plot_fields(medulla_off, res, max_angle, min_angle, cmap, norm)

# plt.figure()
# for neuron in range(split):
#     plt.subplot(2,4,neuron+1)
#     field_2d = np.zeros([len(axis),len(axis)])
#     for i in range(len(axis)):
#         for j in range(len(axis)):
#             field_2d[i,j] = polarity[neuron] * calc_2d_point(axis[i], axis[j], A_rel[neuron], std_cen[neuron], std_sur[neuron])
#     plt.imshow(field_2d, extent=[axis[0],axis[-1],axis[0],axis[-1]],cmap=cmap,norm=norm)
#     plt.colorbar()
#     plt.title(title[neuron])
#     plt.xlabel('Azimuth (deg)')
#     plt.ylabel('Elevation (deg)')
#
#     plt.subplot(2,4,neuron+5)
#     field_1d = np.zeros(len(axis))
#     for i in range(len(axis)):
#         field_1d[i] = polarity[neuron] * calc_1d_point(axis[i], A_rel[neuron], std_cen[neuron], std_sur[neuron])
#
#     plt.plot(axis,field_1d)
#     plt.xlabel('Angle (deg)')
#     plt.ylabel('Response')


plt.show()