from Tuning.tune_neurons_old import tune_neurons
from utilities import save_data, h5_to_dataframe, calc_cap_from_cutoff
import numpy as np

def get_best_params(h5_path, toml_path):
    params_used = np.array([0, 1, 2, 3, 4])
    df_results = h5_to_dataframe(h5_path, toml_path, params_used)
    df_results = df_results.sort_values(by='neglogpost', ascending=True)
    df_results['neglogpost'] = df_results['neglogpost'] - np.min(df_results['neglogpost']) + 1
    df_results_unique = df_results.drop_duplicates()

    best_num = 0
    best = df_results_unique.iloc[best_num]

    return best

h5_t4 = "2023-Mar-24_07-15.t4a.h5"#'2023-Mar-22_22-50.t4a.h5' #"h5_files/g_syns2.h5"
toml_t4 = "conf_t4_reduced.toml"
best_t4 = get_best_params(h5_t4, toml_t4)
h5_t5 = "2023-Mar-25_05-40_t5.h5"#'2023-Mar-22_22-50.t4a.h5' #"h5_files/g_syns2.h5"
toml_t5 = "conf_t5_mcmc.toml"
best_t5 = get_best_params(h5_t5, toml_t5)

cutoff_fast_on = best_t4['Fast Cutoff Freq']
ratio_low_on = best_t4['Ratio Low']
cutoff_ct1_on = best_t4['CT1 Cutoff Freq']
cutoff_mi9 = best_t4['Mi9 Cutoff Freq']
c_inv = best_t4['C Inv']

cutoff_fast_off = best_t5['Fast Cutoff Freq']
ratio_low_off = best_t5['Ratio Low']
cutoff_ct1_off = best_t5['CT1 Cutoff Freq']
g_ct1 = best_t5['CT1 Synaptic Conductance']
rev_ct1 = best_t5['CT1 Synaptic Reversal Potential']

cutoff_retina = max(cutoff_fast_on, cutoff_fast_off)
cutoff_l1_high = cutoff_fast_on
cutoff_l1_low = cutoff_l1_high / ratio_low_on
cutoff_l2_high = cutoff_fast_off
cutoff_l2_low = cutoff_l2_high / ratio_low_off
cutoff_l3 = cutoff_retina
cutoff_mi1 = cutoff_fast_on
cutoff_tm1 = cutoff_fast_off
cutoff_tm9 = cutoff_fast_off
cutoff_t4 = cutoff_fast_on
cutoff_t5 = cutoff_fast_off

cutoffs = [cutoff_retina, cutoff_l1_low, cutoff_l1_high, cutoff_l2_low, cutoff_l2_high, cutoff_l3, cutoff_mi1, cutoff_mi9, cutoff_tm1, cutoff_tm9, cutoff_ct1_on, cutoff_ct1_off, cutoff_t4, cutoff_t4]

fastest = np.max(cutoffs)
dt = min(calc_cap_from_cutoff(fastest)/10, 0.1)
print(dt)
params = tune_neurons(cutoffs, 'all', debug=True, save=False)

params.update({'CInv': c_inv, 'CT1OffG': g_ct1, 'CT1OffReversal': rev_ct1, 'dt': dt,
               'T5': {'name': 'T5', 'type': 'lowpass', 'params': {'cutoff': cutoff_t5, 'invert': False, 'initialValue': 0.0, 'bias': 0.0}}})

save_data(params, 'params_net_10_180.pc')
