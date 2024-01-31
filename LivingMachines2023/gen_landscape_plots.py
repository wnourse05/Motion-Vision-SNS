from typing import Dict
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import wasserstein_distance, energy_distance
import load_conf as lc
import os
import h5py
from matplotlib import cm
from matplotlib.colors import LogNorm
from motion_vision_networks import gen_single_emd_on
from utilities import calc_cap_from_cutoff, h5_to_dataframe
from tqdm import tqdm

# base_folder = Path(r'C:\Users\clayj\Documents\rat_hindlimb')
base_folder = Path("")

mcmc_data = {

    #'0.0': pickle.load(open(base_folder / '2021-May-01_19-50.pypesto_results.0_0-only.true_uniform_prior.40.pkl', 'rb')),
    #'0.1': pickle.load(open(base_folder / '2021-May-04_23-59.pypesto_results.0_1-only.true_uniform_prior.40.pkl', 'rb')),
    #'0.2': pickle.load(open(base_folder / '2021-Apr-30_01-27.pypesto_results.0_2-only.true_uniform_prior.40.pkl', 'rb')),
    #'0.3': pickle.load(open(base_folder / '2021-May-12_04-56.pypesto_results.0_3-only.true_uniform_prior.40.pkl', 'rb')),
    #'0.4': pickle.load(open(base_folder / '2021-May-07_07-34.pypesto_results.0_4-only.true_uniform_prior.40.pkl', 'rb')),
    #'agg': pickle.load(open(base_folder / '2021-May-10_07-18.pypesto_results.all.true_uniform_prior.64.pkl', 'rb')),
}

h5_path = "2023-Mar-24_07-15.t4a.h5"#'2023-Mar-22_22-50.t4a.h5' #"h5_files/g_syns2.h5"
toml_path = "conf_t4_reduced.toml"
# all_list_params = lc.load_param_names(toml_path)
params_used = np.array([0,1,2,3,4])

df_results = h5_to_dataframe(h5_path, toml_path, params_used)

#cleaned_tables = {key: generate_summary_dataframe(value) for key, value in mcmc_data.items()}
#params = list(cleaned_tables['0.0'].columns)[1:] TODO This scales the cost, do this if other option doesn't work
# df_results['neglogpost'] = (df_results['neglogpost'] - 14*np.log(20))/1
df_results['neglogpost'] = df_results['neglogpost'] - np.min(df_results['neglogpost']) + 1

#print({key: table['neglogpost'].min()-9*np.log(10000.0) for key, table in cleaned_tables.items()})
#print({key: table['neglogpost'].min()-9*np.log(10000.0) for key, table in df_results.items()})
#df_results['neglogpost'] = (df_results['neglogpost'] - 14*np.log(20)) / 1e3
print(np.max(df_results['neglogpost']))
print(np.min(df_results['neglogpost']))
print(np.mean(df_results['neglogpost']))
print(2e5)
print(1e0)

df_results = df_results.sort_values(by='neglogpost', ascending=False)
#df_results['neglogpost'] = (df_results['neglogpost'] - np.min(df_results['neglogpost']))/(np.max(df_results['neglogpost']) - np.min(df_results['neglogpost']))
costs = np.unique(df_results['neglogpost'])
# plt.figure()
# plt.hist(df_results['neglogpost'], bins=200)

map_main = plt.cm.magma(np.linspace(0, 0.5, 256))
map_outliers = plt.cm.magma(np.linspace(0.6, 1, 256))
all_colors = np.vstack((map_main, map_outliers))
map_full = mpl.colors.LinearSegmentedColormap.from_list('magma', all_colors)

divnorm = mpl.colors.TwoSlopeNorm(vmin=np.min(df_results['neglogpost']), vcenter=np.mean(df_results['neglogpost']), vmax=np.max(df_results['neglogpost']))

sns.set_style(style='white')
the_cmap = sns.color_palette('magma', as_cmap=True)
norm = LogNorm(np.min(df_results['neglogpost']), np.max(df_results['neglogpost']))

# norm = divnorm

#print(np.where(df_results['neglogpost'] == np.min(df_results['neglogpost'])))

def create_posterior(le_orig_data, filename, sample_ratio:float = 1.0, alpha=0.2, x_axis:str = 'g_Na', y_axis:str = 'g_Kd'):
    sns.reset_defaults()
    # le_data = le_orig_data.sort_values(by=['neglogpost'], ignore_index=True, ascending=False)
    # sample_size = int(sample_ratio * len(le_data['neglogpost']))
    nlp = (le_orig_data['neglogpost'])  # .sample(n=sample_size)
    sns.set_style(style='white')
    g = sns.JointGrid(data=le_orig_data, x=x_axis, y=y_axis, height=10)
    the_cmap = sns.color_palette('Greys_r', as_cmap=True)
    norm = mpl.colors.Normalize(vmin=500, vmax=200000)
    # display_min = max(1e-6, nlp.min()) if nlp.min() < 5.0 else 1e1
    # print(display_min)
    # norm = LogNorm(1e1, 2e3)    # TODO
    norm = LogNorm(1e0, 2e5)
    the_cmap = map_full
    norm = divnorm
    # norm = None
    # inv_norm = LogNorm(1.0/1e3, 1.0/display_min)
    sm = plt.cm.ScalarMappable(cmap=the_cmap, norm=norm)
    # g.plot_joint(sns.scatterplot, alpha=alpha, size=1.0/nlp, sizes=(10, 40), size_norm=inv_norm, hue=nlp, hue_norm=norm, palette=the_cmap, legend=False)  # alpha=0.002, hue_norm=LogNorm(vmin=nlp.min(), vmax=nlp.max()),
    g.plot_joint(sns.scatterplot, alpha=alpha, hue=nlp, hue_norm=norm, palette=the_cmap, legend=False, rasterized=True)  # alpha=0.002, hue_norm=LogNorm(vmin=nlp.min(), vmax=nlp.max()),
    g.ax_joint.figure.colorbar(sm, shrink=10.0, ax=plt.gca())
    g.plot_marginals(sns.histplot, kde=True)
    # plt.savefig(f'posterior_{x_axis}_{y_axis}_{filename}.png', dpi=600)

# print('Making single plot')
# create_posterior(df_results, "old_cost", sample_ratio=1.0, x_axis=list_params[0], y_axis=list_params[1])

def create_posterior_big_plot_color(le_orig_data, filename, sample_ratio: float = 1.0, alpha=0.05):
    sns.set_context("paper", rc={"font.size": 48, "axes.titlesize": 48, "axes.labelsize": 48, "xtick.labelsize": 48,
                                 "ytick.labelsize": 48})
    sns.set_style("white")
    le_data = le_orig_data.sort_values(by=['neglogpost'], ignore_index=True, ascending=False)
    sample_size = int(sample_ratio * len(le_data['neglogpost']))
    nlp = (le_data['neglogpost'])  # .sample(n=sample_size)

    cols_to_use = [col for col in le_data.columns if col != 'neglogpost']
    g = sns.PairGrid(data=le_data, height=10, vars=cols_to_use)
    g.map_offdiag(sns.scatterplot, alpha=alpha, hue=nlp, hue_norm=norm, palette=the_cmap,
                legend=False)
    # g.map_lower(sns.scatterplot, alpha=alpha, hue=nlp, hue_norm=norm, palette=the_cmap,
    #             legend=False)  # alpha=0.002, hue_norm=LogNorm(vmin=nlp.min(), vmax=nlp.max()),
    # g.map_upper(sns.kdeplot, fill=True, thresh=0)#, hue=nlp, hue_norm=norm, palette=the_cmap)
    # g.ax_joint.figure.colorbar(sm, shrink=10.0)
    g.map_diag(sns.histplot, kde=True)
    plt.savefig(f'big_plot_{filename}.png')

# print('Making big plot')
create_posterior_big_plot_color(df_results, "t4_mcmc")



plt.show()
