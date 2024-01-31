import numpy as np
import torch
from motion_vision_networks import gen_single_emd_on
from utilities import cutoff_fastest, gen_gratings, save_data, calc_cap_from_cutoff
from sns_toolbox.renderer import render
import matplotlib.pyplot as plt
import seaborn as sea
from tqdm import tqdm
import time

def gen_stimulus(num_cycles):
    stim_on_lr = np.array([[1.0, 0.0, 0.0],
                           [1.0, 1.0, 0.0],
                           [1.0, 1.0, 1.0],
                           [0.0, 1.0, 1.0],
                           [0.0, 0.0, 1.0],
                           [0.0, 0.0, 0.0]])
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

    # if interval:
    index = 0
    j = 0
    for i in tqdm(range(len(t))):
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
    #         data[i, :] = model(stim[i, :])

    # data = data.to('cpu')
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


    return t, r_l, r_c, r_r, l1_l, l1_c, l1_r, l3_l, l3_c, l3_r, mi1_l, mi1_c, mi1_r, mi9_l, mi9_c, mi9_r, ct1_l, ct1_c,\
        ct1_r, t4_a, t4_b

def plot_emd(dt, interval, stim, cutoff_fast, cutoff_low, cutoff_ct1, cutoff_mi9, c_inv, plot=False, save=True):

    #                   Retina          L1 low    L1 High         L3            Mi1     Mi9          CT1 On          T4     Div gain
    params = np.array([cutoff_fast, cutoff_low, cutoff_fast, cutoff_fast, cutoff_fast, cutoff_mi9, cutoff_ct1, cutoff_fast, 1/c_inv])   # Good guess

    model, net = gen_single_emd_on(dt, params)
    # stim, y = gen_gratings((1,3), freq, 'lr', 5, dt, square=True, device=device)
    # render(net, view=True)
    t, r_l, r_c, r_r, l1_l, l1_c, l1_r, l3_l, l3_c, l3_r, mi1_l, mi1_c, mi1_r, mi9_l, mi9_c, mi9_r, ct1_l, ct1_c,\
            ct1_r, t4_a, t4_b = test_emd(dt, model, net, stim, interval)


    if plot:
        plt.plot(t, t4_a, label='T4a', color='C3')
        plt.plot(t, t4_b, label='T4b', color='C4')
        plt.title('Interval: %.2f ms'%(interval*dt))
        plt.legend()
        sea.despine()
    a_peak = np.max(t4_a)
    b_peak = np.max(t4_b)
    ratio = b_peak/a_peak
    print(a_peak, b_peak, ratio)

    if save:
        trial = {'interval':    interval,
                 't':           t,
                 'r_l':         r_l,
                 'r_c':         r_c,
                 'r_r':         r_r,
                 'l1_l':        l1_l,
                 'l1_c':        l1_c,
                 'l1_r':        l1_r,
                 'l3_l':        l3_l,
                 'l3_c':        l3_c,
                 'l3_r':        l3_r,
                 'mi1_l':       mi1_l,
                 'mi1_c':       mi1_c,
                 'mi1_r':       mi1_r,
                 'mi9_l':       mi9_l,
                 'mi9_c':       mi9_c,
                 'mi9_r':       mi9_r,
                 'ct1_l':       ct1_l,
                 'ct1_c':       ct1_c,
                 'ct1_r':       ct1_r,
                 't4_a':        t4_a,
                 't4_b':        t4_b,
                 'a_peak':      a_peak,
                 'b_peak':      b_peak}

        vel = convert_interval_to_deg_vel(interval, dt)
        param_string = '_%i_%i_%i_%i_%i' % (cutoff_fast, cutoff_low, cutoff_ct1, cutoff_mi9, c_inv)
        filetype = '.pc'
        dir = 'T4 Velocity/'
        filename = dir + str(int(vel)) + '_trial' + param_string + filetype
        # save_data(trial, filename)

    return a_peak, b_peak, ratio

def convert_deg_vel_to_interval(vel, dt):
    scaled_vel = vel/5    # columns per second
    interval = int((1/scaled_vel)/(dt/1000))
    return interval

def convert_interval_to_deg_vel(interval, dt):
    vel = 5000/(interval*dt)
    return vel

def t4_freq_response(stim, vels, params, dt, plot=False, save=True):
    cutoff_fast = params[0]
    cutoff_low = params[1]
    cutoff_ct1 = params[2]
    cutoff_mi9 = params[3]
    c_inv = params[4]

    a_peaks = np.zeros(len(vels))
    b_peaks = np.zeros_like(a_peaks)
    ratios = np.zeros_like(a_peaks)
    dpi = 100
    size = (9,6)
    if plot:
        dir = 'T4 Velocity/'
        fig = plt.figure(figsize=size, dpi=dpi)
        plt.suptitle('Cutoff_fast: %.2f, Ratio_low: %.2f, Ratio_CT1: %.2f, Ratio_Mi9: %.2f, C_inv: %.2f' % (
        cutoff_fast, cutoff_low, cutoff_ct1, cutoff_mi9, c_inv))
    for i in range(len(vels)):
        # print(vels[i])
        if plot:
            plt.subplot(len(vels),1,i+1)
        interval = convert_deg_vel_to_interval(vels[i], dt)
        # print(interval)
        a_peaks[i], b_peaks[i], ratios[i] = plot_emd(dt, interval, stim, cutoff_fast, cutoff_low, cutoff_ct1, cutoff_mi9, c_inv, plot=plot, save=save)
    param_string = '_%i_%i_%i_%i_%i'%(cutoff_fast, cutoff_low, cutoff_ct1, cutoff_mi9, c_inv)
    if plot:
        filetype = '.svg'
        # plt.savefig(dir+'data'+param_string+filetype, dpi=dpi)

        fig1 = plt.figure(figsize=size, dpi=dpi)
        # print(fig1.dpi)
        plt.suptitle('Cutoff_fast: %.2f, Ratio_low: %.2f, Cutoff_CT1: %.2f, Cutoff_Mi9: %.2f, C_inv: %.2f'%(cutoff_fast, cutoff_low, cutoff_ct1, cutoff_mi9, c_inv))
        print(b_peaks)
        print(a_peaks)
        print(ratios)
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
        # plt.savefig(dir+'curve' + param_string + filetype, dpi=dpi)
    if save:
        dir = 'T4 Velocity/'
        data = {'vels':     vels,
                'a_peaks':  a_peaks,
                'b_peaks':  b_peaks,
                'ratios':   ratios}
        filename = dir + 'set' + param_string + '.pc'
        # save_data(data, filename)

def init():
    sea.set_theme()
    sea.set_style('ticks')
    sea.color_palette('colorblind')
    stim_on_lr = gen_stimulus(10)
    num_intervals = 4
    vels = np.linspace(10,180,num=num_intervals)
    vels = np.array([9,10,11,20,50,100,150,200,250,300,360,720])
    v_slow = 10
    v_fast = 360
    dt = 0.1
    goal = np.linspace(1.0, 0.1, num=num_intervals)
    wavelength = 30
    return stim_on_lr, vels, dt, goal, v_slow, v_fast, wavelength

stim, vels, dt, goal, v_slow, v_fast, wavelength = init()
cap_fast = 10*dt
cutoff_fast = calc_cap_from_cutoff(cap_fast)
cap_low = 100*wavelength/v_fast
cutoff_low = calc_cap_from_cutoff(cap_low)
cutoff_ct1 = cutoff_fast
cap_mi9 = 1000*5/(5*v_slow)
cutoff_mi9 = calc_cap_from_cutoff(cap_mi9)
# cutoff_mi9 = 1
c_inv=10
# params = [200, 10, 200, cutoff_mi9, 10]
params = [cutoff_fast, cutoff_low, cutoff_ct1, cutoff_mi9, c_inv]
print(params)
start = time.time()
t4_freq_response(stim, vels, params, dt, plot=True, save=True)
end = time.time()-start
print(end)

plt.show()
print('Done')
