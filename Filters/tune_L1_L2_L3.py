import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import signal

def calc_1d_point(x, A_rel, std_cen, std_sur):
    return np.exp((-x**2)/(2*std_cen**2)) - A_rel*np.exp((-x**2)/(2*std_sur**2))

def adjust_example(data, index, R, rest=0, res=5, min_angle=-20, max_angle=20):
    fig = plt.figure()

    # Original Receptive Field
    axis = np.arange(min_angle, max_angle + res, res)
    field_1d = np.zeros(len(axis))
    for i in range(len(axis)):
        field_1d[i] = data['polarity'][index] * calc_1d_point(axis[i], data['A_rel'][index], data['std_cen'][index],
                                                               data['std_sur'][index])
    plt.subplot(2,2,1)
    plt.plot(axis, field_1d)
    plt.xlabel('Angle (deg)')
    plt.ylabel('Response')
    plt.title('Original Spatial Response')

    # Scaled Receptive Field
    max_val = np.max(field_1d)
    min_val = np.min(field_1d)
    diff = max_val - min_val
    field_1d_norm = field_1d/diff
    field_1d_scaled = field_1d_norm*R
    new_rest = rest-np.min(field_1d_scaled)
    field_1d_shifted = field_1d_scaled + new_rest
    # print(np.min(field_1d_shifted))
    # print(np.max(field_1d_shifted))
    plt.subplot(2,2,2)
    plt.plot(axis, field_1d_shifted)
    plt.xlabel('Angle (deg)')
    plt.ylabel('Response')
    plt.title('Scaled Spatial Response')

    # Original Step Response
    lowcut = 1 / (data['tau_lp'][index])
    if data['tau_hp'][index] is None:
        # print(lowcut)
        b, a = signal.butter(1, lowcut, 'low', analog=True)
    else:
        highcut = 1 / (data['tau_hp'][index])
        # print(lowcut, highcut)
        b, a = signal.butter(1, [highcut, lowcut], 'band', analog=True)
    t, y = signal.step((b, a))
    plt.subplot(2,2,3)
    plt.plot(t, y * data['polarity'][index], label='Response')

    # plt.legend()
    plt.xlabel('t')
    plt.ylabel('Response')
    plt.title('Original Step Response')

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
    print(np.min(y_shifted), np.max(y_shifted))
    print(rest-np.min(field_1d_scaled))
    plt.subplot(2,2,4)
    plt.plot(t,y_shifted)
    plt.xlabel('t')
    plt.ylabel('Response')
    plt.title('Scaled Step Response')

    return fig


lamina = {'title':      ['L1', 'L2', 'L3', 'L4', 'L5'],
          'A_rel':      [0.012, 0.013, 0.193, 0.046, 0.035],
          'std_cen':    [2.679, 2.861, 2.514, 3.633, 2.883],
          'std_sur':    [17.473, 12.449, 6.423, 13.911, 13.325],
          'polarity':   [-1, -1, -1, -1, 1],
          'tau_hp':     [0.391, 0.288, None,  0.381, 0.127],
          'tau_lp':     [0.038, 0.058, 0.054, 0.023, 0.042]}

figL1 = adjust_example(lamina,0,20)
figL2 = adjust_example(lamina,1,20)
figL3 = adjust_example(lamina,2,20)
plt.show()
