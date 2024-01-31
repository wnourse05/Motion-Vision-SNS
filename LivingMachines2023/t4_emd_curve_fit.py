import numpy as np
from motion_vision_networks import gen_single_emd_on
from utilities import cutoff_fastest, gen_gratings, save_data, calc_cap_from_cutoff
import matplotlib.pyplot as plt
import seaborn as sea
from tqdm import tqdm
from scipy.optimize import minimize

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
    # r_l = data_sns_toolbox[0, :]
    # r_c = data_sns_toolbox[1, :]
    # r_r = data_sns_toolbox[2, :]
    # l1_l = data_sns_toolbox[3, :]
    # l1_c = data_sns_toolbox[4, :]
    # l1_r = data_sns_toolbox[5, :]
    # l3_l = data_sns_toolbox[6, :]
    # l3_c = data_sns_toolbox[7, :]
    # l3_r = data_sns_toolbox[8, :]
    # mi1_l = data_sns_toolbox[9, :]
    # mi1_c = data_sns_toolbox[10, :]
    # mi1_r = data_sns_toolbox[11, :]
    # mi9_l = data_sns_toolbox[12, :]
    # mi9_c = data_sns_toolbox[13, :]
    # mi9_r = data_sns_toolbox[14, :]
    # ct1_l = data_sns_toolbox[15, :]
    # ct1_c = data_sns_toolbox[16, :]
    # ct1_r = data_sns_toolbox[17, :]
    t4_a = data[18, :]
    t4_b = data[19, :]

    a_peak = np.max(t4_a)
    b_peak = np.max(t4_b)
    ratio = b_peak / a_peak



    return a_peak, b_peak, ratio, t4_a, t4_b, t

def freq_response_emd(intervals, num_intervals, stim, params_mcmc):
    cutoff_fast = params_mcmc[0]
    ratio_low = params_mcmc[1]
    cutoff_ct1 = cutoff_fast
    cutoff_mi9 = params_mcmc[2]
    c_inv = params_mcmc[3]
    dt = calc_cap_from_cutoff(cutoff_fast)/10

    #                   Retina          L1 low              L1 High         L3              Mi1     Mi9          CT1 On          T4     Div gain
    params = np.array([cutoff_fast, cutoff_fast/ratio_low, cutoff_fast, cutoff_fast, cutoff_fast, cutoff_mi9, cutoff_ct1, cutoff_fast, 1/c_inv])   # Good guess

    model, net = gen_single_emd_on(dt, params)

    a_peaks = np.zeros_like(intervals)
    b_peaks = np.zeros_like(intervals)
    ratios = np.zeros_like(intervals)

    for i in tqdm(range(num_intervals)):
        # plt.figure()
        a_peaks[i], b_peaks[i], ratios[i], t4_a, t4_b, t = test_emd(dt, model, net, stim, intervals[i])
        # plt.plot(t, t4_a)
        # plt.plot(t, t4_b)
    # a_peaks = np.clip(a_peaks, -100, 100)
    # b_peaks = np.clip(b_peaks, -100, 100)
    # ratios = np.clip(ratios, -100, 100)

    return a_peaks, b_peaks, ratios

def sum_of_lsq(goal, data):
    error_vector = (goal-data)**2
    error_lsq = np.sum(error_vector)
    return error_lsq

def cost_function(a_peaks, b_peaks, ratios, goal):
    cost = 100*sum_of_lsq(b_peaks, goal)
    if np.isnan(cost):
        cost =1e5

    return cost

def evaluate(params, stim, intervals, goal):
    # print(params)
    a_peaks, b_peaks, ratios = freq_response_emd(intervals, 4, stim, params)
    cost = cost_function(a_peaks, b_peaks, ratios, goal)
    print(params, cost)
    return cost


sea.set_theme(style='ticks')
sea.set_palette('colorblind')
stim_on_lr = gen_stimulus(1)
num_intervals = 4
dt = 0.1
vels = np.linspace(10, 180, num=num_intervals)
intervals = convert_deg_vel_to_interval(vels, dt)
# goal = np.linspace(1.0, 0.1, num=num_intervals)
goal_x = np.linspace(0.0, 4.0, num=num_intervals)
goal = (4*np.exp(goal_x))/(1+np.exp(goal_x))**2
x0 = np.array([200.0, 10.0, 5.0, 10.0])

# a_peaks, b_peaks, ratios = freq_response_emd(dt, intervals, num_intervals, stim_on_lr, p0)
#
# plt.figure()
# plt.plot(vels, a_peaks)
# plt.plot(vels, b_peaks)

f = lambda x: evaluate(x, stim_on_lr, intervals, goal)
bound_cutoff_low = 0.1
bound_cutoff_high = 300
bound_ratio_low = 1
bound_ratio_high = 100
bound_gain_low = 1
bound_gain_high = 100
bounds = ((bound_cutoff_low, bound_cutoff_high), (bound_ratio_low, bound_ratio_high), (bound_cutoff_low, bound_cutoff_high), (bound_gain_low, bound_gain_high))
print('Starting Optimization')
result = minimize(f, x0, method='L-BFGS-B', bounds=bounds, options={'disp': True})
params = result.x
print('Final params:')
print(params)

save_data(params, 't4_params.p')

# num_more = 50
# vels_more = np.linspace(10,360, num=num_more)
# intervals_more = convert_deg_vel_to_interval(vels_more, dt)
# a_peaks, b_peaks, ratios = freq_response_emd(dt, intervals_more, num_more, stim_on_lr, params)
#
#
# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(vels_more, a_peaks)
# plt.plot(vels_more, b_peaks)
# # plt.plot(vels, goal)
# plt.xscale('log')
# plt.subplot(2,1,2)
# plt.plot(ratios)
# plt.xscale('log')
#
# plt.show()