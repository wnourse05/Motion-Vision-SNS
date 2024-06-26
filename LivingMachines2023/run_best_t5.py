from utilities import save_data, h5_to_dataframe, calc_cap_from_cutoff
from motion_vision_networks import gen_single_emd_off
import numpy as np
from tqdm import tqdm

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

def convert_deg_vel_to_interval(vel, dt):
    scaled_vel = vel/5    # columns per second
    interval = ((1/scaled_vel)/(dt/1000))
    return interval

def test_emd(dt, model, net, stim, interval):
    model.reset()
    interval = int(interval)
    num_samples = np.shape(stim)[0]
    t = np.linspace(0, dt*num_samples*interval, num=num_samples*int(interval))

    data = np.zeros([num_samples*int(interval), net.get_num_outputs_actual()])

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
    #         data_sns_toolbox[i, :] = model_toolbox(stim[i, :])

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


    # trial = {'interval':    interval,
    #          't':           t,
    #          'r_l':         r_l,
    #          'r_c':         r_c,
    #          'r_r':         r_r,
    #          'l1_l':        l1_l,
    #          'l1_c':        l1_c,
    #          'l1_r':        l1_r,
    #          'l3_l':        l3_l,
    #          'l3_c':        l3_c,
    #          'l3_r':        l3_r,
    #          'mi1_l':       mi1_l,
    #          'mi1_c':       mi1_c,
    #          'mi1_r':       mi1_r,
    #          'mi9_l':       mi9_l,
    #          'mi9_c':       mi9_c,
    #          'mi9_r':       mi9_r,
    #          'ct1_l':       ct1_l,
    #          'ct1_c':       ct1_c,
    #          'ct1_r':       ct1_r,
    #          't4_a':        t4_a,
    #          't4_b':        t4_b,
    #          'a_peak':      a_peak,
    #          'b_peak':      b_peak}

    # vel_index = convert_interval_to_deg_vel(interval, dt)
    # param_string = '_%i_%i_%i_%i_%i' % (cutoff_fast, ratio_low, cutoff_ct1, cutoff_mi9, c_inv)
    # filename = dir + str(int(vel_index)) + '_trial' + param_string
    # save_data(trial, filename)

    return a_peak, b_peak, ratio

def freq_response_emd(vels, num_intervals, stim, cutoff_fast, ratio_low, cutoff_ct1, g_ct1, rev_ct1, ranking):
    #                   Retina        L2 low                  L2 High      L3               Tm1         Tm9        CT1 Off        T5        g    reversal
    params = np.array([cutoff_fast, cutoff_fast / ratio_low, cutoff_fast, cutoff_fast, cutoff_fast, cutoff_fast, cutoff_ct1, cutoff_fast, g_ct1, rev_ct1])  # Good guess
    # print(params)
    # dt = min(calc_cap_from_cutoff(cutoff_fast) / 10, 0.1)
    fast = max(cutoff_fast, cutoff_ct1)
    dt = min(calc_cap_from_cutoff(fast)/10, 0.1)
    print(dt)
    intervals = convert_deg_vel_to_interval(vels, dt)

    model, net = gen_single_emd_off(dt, params)

    # param_string = '_%i_%i_%i_%i_%i.pc' % (cutoff_fast, ratio_low, cutoff_ct1, cutoff_mi9, c_inv)
    # dir = 'T4 Velocity/'

    a_peaks = np.zeros_like(intervals)
    b_peaks = np.zeros_like(intervals)
    ratios = np.zeros_like(intervals)

    for i in range(num_intervals):
        print('\n     Interval %i/%i'%(i+1, num_intervals))
        a_peaks[i], b_peaks[i], ratios[i] = test_emd(dt, model, net, stim, intervals[i])

    return a_peaks, b_peaks, ratios

num_intervals = 50
goal_x = np.linspace(0.0, 4.0, num=num_intervals)
goal = (4 * np.exp(goal_x)) / (1 + np.exp(goal_x)) ** 2
vels = np.linspace(10, 180, num=num_intervals)
stim_on_lr = gen_stimulus(1)

h5_path = "2023-Mar-25_05-40_t5.h5"#'2023-Mar-22_22-50.t4a.h5' #"h5_files/g_syns2.h5"
toml_path = "conf_t5_mcmc.toml"
# all_list_params = lc.load_param_names(toml_path)
params_used = np.array([0,1,2,3,4])

df_results = h5_to_dataframe(h5_path, toml_path, params_used)
df_results = df_results.sort_values(by='neglogpost', ascending=True)
df_results['neglogpost'] = df_results['neglogpost'] - np.min(df_results['neglogpost']) + 1
df_results_unique = df_results.drop_duplicates()

best_num = 10
best = df_results_unique.iloc[:best_num]
cutoff_fast = best['Fast Cutoff Freq']
ratio_low = best['Ratio Low']
cutoff_ct1 = best['CT1 Cutoff Freq']
g_ct1 = best['CT1 Synaptic Conductance']
rev_ct1 = best['CT1 Synaptic Reversal Potential']

data = {'vels': vels,
        'goalResponse': goal,
        'stimulus': stim_on_lr,
        'best': best}

for i in (range(best_num)):
    print('Number %i/%i'%(i+1, best_num))
    a_peaks, b_peaks, ratios = freq_response_emd(vels, num_intervals, stim_on_lr, cutoff_fast.iloc[i], ratio_low.iloc[i], cutoff_ct1.iloc[i], g_ct1.iloc[i], rev_ct1.iloc[i], i)
    trial = {'aPeaks': a_peaks,
             'bPeaks': b_peaks,
             'ratios': ratios}
    name = 'trial%i'%i
    data[name] = trial

save_data(data, 't5_best_results.pc')
