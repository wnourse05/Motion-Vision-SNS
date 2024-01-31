import numpy as np
from motion_vision_networks import gen_single_emd_on
from utilities import cutoff_fastest, gen_gratings, save_data, calc_cap_from_cutoff, load_data
import matplotlib.pyplot as plt
import seaborn as sea
from tqdm import tqdm
from scipy.special import expit

def convert_deg_vel_to_interval(vel, dt):
    scaled_vel = vel/5    # columns per second
    interval = ((1/scaled_vel)/(dt/1000))
    return interval

def convert_interval_to_deg_vel(interval, dt):
    vel = 5000/(interval*dt)
    return vel

def gen_stimulus(num_cycles):
    stim_on_lr = np.array([[1.0, 0.0, 0.0],
                           [1.0, 1.0, 0.0],
                           [1.0, 1.0, 1.0]])
    for i in range(num_cycles):
        if i == 0:
            stim_full = np.copy(stim_on_lr)
        else:
            stim_full = np.vstack((stim_on_lr, stim_full))
    return stim_full

def test_emd(dt, model, net, stim, interval):
    model.reset()
    interval = int(interval)
    num_samples = np.shape(stim)[0]
    t = np.linspace(0, dt*num_samples*interval, num=num_samples*interval)

    data = np.zeros([num_samples*interval, net.get_num_outputs_actual()])

    # if interval:
    index = 0
    j = 0
    for i in (range(len(t))):
        if index < num_samples:
            data[i,:] = model(stim[index,:])
        else:
            data[i, :] = model(stim[-1, :])
        j += 1
        if j == interval:
            index += 1
            j = 0
    # else:
    #     for i in range(len(t)):
    #         data_sns_toolbox[i, :] = model(stim[i, :])

    # data_sns_toolbox = data_sns_toolbox.to('cpu')
    data = data.transpose()
    r_l = data[0, :]
    r_c = data[1, :]
    r_r = data[2, :]
    l1_l = data[3, :]
    l1_c = data[4, :]
    l1_r = data[5, :]
    l3_l = data[6, :]
    l3_c = data[7, :]
    l3_r = data[8, :]
    mi1_l = data[9, :]
    mi1_c = data[10, :]
    mi1_r = data[11, :]
    mi9_l = data[12, :]
    mi9_c = data[13, :]
    mi9_r = data[14, :]
    ct1_l = data[15, :]
    ct1_c = data[16, :]
    ct1_r = data[17, :]
    t4_a = data[18, :]
    t4_b = data[19, :]

    a_peak = np.max(t4_a)
    b_peak = np.max(t4_b)
    ratio = b_peak / a_peak
    plt.subplot(2,1,1)
    plt.plot(t, t4_a, label=a_peak)
    plt.subplot(2,1,2)
    plt.plot(t, t4_b, label=b_peak)



    return a_peak, b_peak, ratio, t4_a, t4_b, t

def freq_response_emd(dt, intervals, num_intervals, stim, params_mcmc):
    cutoff_fast = params_mcmc[0]
    ratio_low = params_mcmc[1]
    cutoff_ct1 = params_mcmc[2]
    cutoff_mi9 = params_mcmc[3]
    c_inv = params_mcmc[4]

    #                   Retina          L1 low              L1 High         L3              Mi1     Mi9          CT1 On          T4     Div gain
    params = np.array([cutoff_fast, cutoff_fast/ratio_low, cutoff_fast, cutoff_fast, cutoff_fast, cutoff_mi9, cutoff_ct1, cutoff_fast, 1/c_inv])   # Good guess
    print(params)

    model, net = gen_single_emd_on(dt, params)

    a_peaks = np.zeros_like(intervals)
    b_peaks = np.zeros_like(intervals)
    ratios = np.zeros_like(intervals)
    plt.figure()
    for i in tqdm(range(num_intervals)):
        # plt.figure()
        a_peaks[i], b_peaks[i], ratios[i], t4_a, t4_b, t = test_emd(dt, model, net, stim, intervals[i])
        # plt.plot(t, t4_a)
        # plt.plot(t, t4_b)
    plt.subplot(2,1,1)
    plt.legend()
    plt.subplot(2,1,2)
    plt.legend()


    return a_peaks, b_peaks, ratios

def sum_of_lsq(goal, data):
    error_vector = (goal-data)**2
    error_lsq = np.sum(error_vector)
    return error_lsq


params = load_data('t4_params.p')
params = np.array([200, 10, 200, 200, 10])
print(params)

sea.set_theme(style='ticks')
sea.set_palette('colorblind')
stim_on_lr = gen_stimulus(1)
dt = min(calc_cap_from_cutoff(params[0])/10, 0.1)
print(dt)
num_intervals = 4
# dt = 0.1
vels = np.linspace(10, 180, num=num_intervals)
intervals = convert_deg_vel_to_interval(vels, dt)
# goal = np.linspace(1.0, 0.1, num=num_intervals)
goal_x = np.linspace(0,4, num=num_intervals)
goal = (4 * np.exp(goal_x)) / (1 + np.exp(goal_x)) ** 2

a_peaks, b_peaks, ratios = freq_response_emd(dt, intervals, num_intervals, stim_on_lr, params)

print(100*sum_of_lsq(b_peaks, goal))
plt.figure()
plt.subplot(2,1,1)
plt.plot(vels, a_peaks, label='A Peaks')
plt.plot(vels, b_peaks, label='B Peaks')
plt.xscale('log')
plt.legend()
plt.subplot(2,1,2)
plt.plot(vels, ratios)
plt.xscale('log')

# x = np.linspace(-3,9)
# y = expit(-x/2)
# scale = 1/y[0]
# # y *= scale
#
# x_der = np.linspace(0,6)
# y_der = (4*np.exp(x_der))/(1+np.exp(x_der))**2
#
# num_more = 50
# vels_more = np.linspace(10,360, num=num_more)
# intervals_more = convert_deg_vel_to_interval(vels_more, dt)
# a_peaks, b_peaks, ratios = freq_response_emd(dt, intervals_more, num_more, stim_on_lr, params)
#
#
# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(vels_more, a_peaks, label='A Peaks')
# plt.plot(vels_more, b_peaks, label='B Peaks')
# plt.plot(vels_more, np.linspace(1,0.1, num=num_more), label='linear')
# plt.plot(vels_more, y, label='logistic')
# plt.plot(vels_more, y_der, label='der logistic')
# plt.legend()
# # plt.xscale('log')
# plt.subplot(2,1,2)
# plt.plot(vels_more, ratios)
# plt.xscale('log')

plt.show()