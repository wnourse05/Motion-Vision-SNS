import numpy as np
import torch
from motion_vision_networks import gen_single_emd_off
from utilities import cutoff_fastest, gen_gratings, save_data, calc_cap_from_cutoff, synapse_target, activity_range
from sns_toolbox.renderer import render
import matplotlib.pyplot as plt
import seaborn as sea
from tqdm import tqdm
import time

def gen_stimulus(num_cycles):
    stim_on_lr = np.array([[1.0, 1.0, 1.0],
                           [0.0, 1.0, 1.0],
                           [0.0, 0.0, 1.0],
                           [0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0],
                           [1.0, 1.0, 0.0]])
    for i in range(num_cycles):
        if i == 0:
            stim_full = np.copy(stim_on_lr)
        else:
            stim_full = np.vstack((stim_on_lr, stim_full))
    return stim_full
def test_emd(dt, model, net, stim, interval):
    model.reset()

    num_samples = np.shape(stim)[0]
    t = np.linspace(0, dt*num_samples*interval, num=num_samples*interval)

    data = np.zeros([num_samples*interval, net.get_num_outputs_actual()])
    stim_l = np.zeros_like(t)
    stim_c = np.zeros_like(t)
    stim_r = np.zeros_like(t)

    # if interval:
    index = 0
    j = 0
    for i in tqdm(range(len(t))):
        if index < num_samples:
            data[i,:] = model(stim[index,:])
            stim_l[i] = stim[index, 0]
            stim_c[i] = stim[index, 1]
            stim_r[i] = stim[index, 2]
        else:
            data[i, :] = model(stim[-1, :])
            stim_l[i] = stim[-1, 0]
            stim_c[i] = stim[-1, 1]
            stim_r[i] = stim[-1, 2]
        j += 1
        if j == interval:
            index += 1
            j = 0
    # else:
    #     for i in range(len(t)):
    #         data[i, :] = model(stim[i, :])

    # data = data.to('cpu')
    data = data.transpose()
    r_l = data[0, :]
    r_c = data[1, :]
    r_r = data[2, :]
    l2_l = data[3, :]
    l2_c = data[4, :]
    l2_r = data[5, :]
    l3_l = data[6, :]
    l3_c = data[7, :]
    l3_r = data[8, :]
    tm1_l = data[9, :]
    tm1_c = data[10, :]
    tm1_r = data[11, :]
    tm9_l = data[12, :]
    tm9_c = data[13, :]
    tm9_r = data[14, :]
    ct1_l = data[15, :]
    ct1_c = data[16, :]
    ct1_r = data[17, :]
    t5_a = data[18, :]
    t5_b = data[19, :]


    return t, r_l, r_c, r_r, l2_l, l2_c, l2_r, l3_l, l3_c, l3_r, tm1_l, tm1_c, tm1_r, tm9_l, tm9_c, tm9_r, ct1_l, ct1_c,\
        ct1_r, stim_l, stim_c, stim_r, t5_a, t5_b

def plot_emd(dt, interval, stim, params, plot=False, debug=False):

    #                        Retina          L2 low          L2 High      L3         Tm1     Tm9          CT1 Off        T4     g           reversal
    params_full = np.array([params[0], params[0]/params[1], params[0], params[0], params[0], params[0], params[2], params[0], params[3], params[4]])   # Good guess

    model, net = gen_single_emd_off(dt, params_full)
    # stim, y = gen_gratings((1,3), freq, 'lr', 5, dt, square=True, device=device)
    # render(net, view=True)
    t, r_l, r_c, r_r, l2_l, l2_c, l2_r, l3_l, l3_c, l3_r, tm1_l, tm1_c, tm1_r, tm9_l, tm9_c, tm9_r, ct1_l, ct1_c,\
            ct1_r, stim_l, stim_c, stim_r, t5_a, t5_b = test_emd(dt, model, net, stim, interval)


    a_peak = np.max(t5_a)
    b_peak = np.max(t5_b)
    ratio = b_peak/a_peak

    if debug:
        plt.figure()
        plt.suptitle('Interval: %.2f ms'%(interval*dt))
        plt.subplot(5,1,1)
        plt.title('Stim')
        plt.plot(t, stim_l, label='left')
        plt.plot(t, stim_c, label='center')
        plt.plot(t, stim_r, label='right')
        plt.legend()
        sea.despine()
        plt.subplot(5,1,2)
        plt.title('Lamina')
        plt.plot(t, l2_l, label='left')
        plt.plot(t, l2_c, label='center')
        plt.plot(t, l2_r, label='right')
        plt.legend()
        sea.despine()
        plt.subplot(5, 1, 3)
        plt.title('PD')
        plt.plot(t, tm9_l, label='Tm9', color='C3')
        plt.plot(t, tm1_c, label='Tm1', color='C4')
        plt.plot(t, ct1_r, label='CT1 Off', color='C5')
        plt.legend()
        sea.despine()
        plt.subplot(5, 1, 4)
        plt.title('ND')
        plt.plot(t, ct1_l, label='CT1 Off', color='C5')
        plt.plot(t, tm1_c, label='Tm1', color='C4')
        plt.plot(t, tm9_r, label='Tm9', color='C3')
        plt.legend()
        sea.despine()
        plt.subplot(5, 1, 5)
        plt.plot(t, t5_a, color='C6', label='T5_a')
        plt.plot(t, t5_b, color='C7', label='T5_b')
        plt.legend()
        sea.despine()
    elif plot:
        plt.title('Interval: %.2f ms' % (interval * dt))
        plt.plot(t, t5_a, label='T5_a')
        plt.plot(t, t5_b, label='T5_b')
        plt.legend()
        sea.despine()

    return a_peak, b_peak, ratio


def convert_deg_vel_to_interval(vel, dt):
    scaled_vel = vel/5    # columns per second
    interval = int((1/scaled_vel)/(dt/1000))
    return interval

def convert_interval_to_deg_vel(interval, dt):
    vel = 5000/(interval*dt)
    return vel

def t5_freq_response(dt, vels, stim, params, plot=False):
    a_peaks = np.zeros_like(vels)
    b_peaks = np.zeros_like(vels)
    ratios = np.zeros_like(vels)
    if plot:
        fig = plt.figure()
    for i in range(len(vels)):
        # print(vels[i])
        if plot:
            plt.subplot(len(vels),1,i+1)
        interval = convert_deg_vel_to_interval(vels[i], dt)
        # print(interval)
        a_peaks[i], b_peaks[i], ratios[i] = plot_emd(dt, interval, stim, params, plot=plot)
    if plot:
        fig1 = plt.figure()
        # print(fig1.dpi)
        plt.subplot(2,1,1)
        plt.plot(vels, a_peaks)
        plt.plot(vels, b_peaks)
        sea.despine()
        plt.xscale('log')
        plt.title('B Peak')
        plt.subplot(2,1,2)
        plt.plot(vels, ratios)
        plt.title('B/A')
        plt.xscale('log')
        sea.despine()

sea.set_theme()
sea.set_style('ticks')
sea.color_palette('colorblind')
stim_off_lr = gen_stimulus(2)
num_intervals = 4
vels = np.linspace(10,720,num=num_intervals)
#
dt = 0.1
goal = np.linspace(1.0, 0.1, num=num_intervals)

g_pd_t5, rev_pd_t5 = synapse_target(0.0, activity_range)
params = [200, 10, 10, g_pd_t5, 0.0]

t5_freq_response(dt, vels, stim_off_lr, params, plot=True)
# start = time.time()
interval = 100
# plot_emd(dt, interval, stim_off_lr, params, plot=False, debug=True)
# end = time.time()-start
# print(end)

plt.show()
