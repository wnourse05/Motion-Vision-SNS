from typing import Dict
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance, energy_distance
import load_conf as lc
import os
import h5py
from matplotlib import cm
from matplotlib.colors import LogNorm

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

h5_path = base_folder / "2023-Mar-17_20-t4.h5" #"h5_files/g_syns2.h5"
toml_path = Path("conf_t4.toml")
all_list_params = lc.load_param_names(toml_path)
params_used = np.array([0,1,2,3,4,5,6,7,8,9,10,11])
list_params = []

for i in params_used:
    list_params.append(all_list_params[i])

with h5py.File(h5_path) as h5_file:
    trace_x = h5_file['trace_x'][()]  # ex. (24, 12001 20)
    print(trace_x.shape)
    trace_neglogpost = h5_file['trace_neglogpost'][()]  # ex. (24, 12001)
    print(trace_neglogpost.shape)
    trace_neglogpost_flat = trace_neglogpost.flatten()
    print(trace_neglogpost_flat.shape)
    trace_x_flat = trace_x.reshape((trace_x.shape[0] * trace_x.shape[1], trace_x.shape[2]))
    sr_trace_neglogpost = pd.Series(data=trace_neglogpost_flat, name='neglogpost')
    df_results = pd.DataFrame(data=trace_x_flat, columns=list_params)
    df_results.insert(0, "neglogpost", sr_trace_neglogpost)

#cleaned_tables = {key: generate_summary_dataframe(value) for key, value in mcmc_data.items()}
#params = list(cleaned_tables['0.0'].columns)[1:] TODO This scales the cost, do this if other option doesn't work
df_results['neglogpost'] = (df_results['neglogpost'] - 14*np.log(20))/1
df_results['neglogpost'] = df_results['neglogpost'] - np.min(df_results['neglogpost']) + 1

#print({key: table['neglogpost'].min()-9*np.log(10000.0) for key, table in cleaned_tables.items()})
#print({key: table['neglogpost'].min()-9*np.log(10000.0) for key, table in df_results.items()})
#df_results['neglogpost'] = (df_results['neglogpost'] - 14*np.log(20)) / 1e3
print(np.max(df_results['neglogpost']))
print(np.min(df_results['neglogpost']))
print(np.mean(df_results['neglogpost']))
print(2e3)
print(1e0)

df_results = df_results.sort_values(by='neglogpost', ascending=False)
#df_results['neglogpost'] = (df_results['neglogpost'] - np.min(df_results['neglogpost']))/(np.max(df_results['neglogpost']) - np.min(df_results['neglogpost']))

print(df_results)
#print(np.where(df_results['neglogpost'] == np.min(df_results['neglogpost'])))

def create_posterior(le_orig_data, filename, sample_ratio:float = 1.0, alpha=0.2, x_axis:str = 'g_Na', y_axis:str = 'g_Kd'):
    sns.reset_defaults()
    # le_data = le_orig_data.sort_values(by=['neglogpost'], ignore_index=True, ascending=False)
    # sample_size = int(sample_ratio * len(le_data['neglogpost']))
    nlp = (le_orig_data['neglogpost'])  # .sample(n=sample_size)
    sns.set_style(style='white')
    g = sns.JointGrid(data=le_orig_data, x=x_axis, y=y_axis, height=10)
    the_cmap = cm.get_cmap(name='plasma')
    # display_min = max(1e-6, nlp.min()) if nlp.min() < 5.0 else 1e1
    # print(display_min)
    norm = LogNorm(1e1, 2e3)    # TODO
    #norm = None
    # inv_norm = LogNorm(1.0/1e3, 1.0/display_min)
    sm = plt.cm.ScalarMappable(cmap=the_cmap, norm=norm)
    # g.plot_joint(sns.scatterplot, alpha=alpha, size=1.0/nlp, sizes=(10, 40), size_norm=inv_norm, hue=nlp, hue_norm=norm, palette=the_cmap, legend=False)  # alpha=0.002, hue_norm=LogNorm(vmin=nlp.min(), vmax=nlp.max()),
    g.plot_joint(sns.scatterplot, alpha=alpha, hue=nlp, hue_norm=norm, palette=the_cmap, legend=False, rasterized=True)  # alpha=0.002, hue_norm=LogNorm(vmin=nlp.min(), vmax=nlp.max()),
    g.ax_joint.figure.colorbar(sm, shrink=10.0)
    g.plot_marginals(sns.histplot, kde=True)
    plt.savefig(f'posterior_{x_axis}_{y_axis}_{filename}.png', dpi=600)

print('Making first plot')
create_posterior(df_results, "old_cost", sample_ratio=1.0, x_axis=list_params[5], y_axis=list_params[3])

print('Making second plot')
def create_posterior_big_plot_color(le_orig_data, filename, sample_ratio: float = 1.0, alpha=0.05):
    sns.set_context("paper", rc={"font.size": 48, "axes.titlesize": 48, "axes.labelsize": 48, "xtick.labelsize": 48,
                                 "ytick.labelsize": 48})
    sns.set_style("white")
    plt.ioff()
    le_data = le_orig_data.sort_values(by=['neglogpost'], ignore_index=True, ascending=False)
    sample_size = int(sample_ratio * len(le_data['neglogpost']))
    nlp = (le_data['neglogpost'])  # .sample(n=sample_size)
    print(nlp.min())
    the_cmap = cm.get_cmap(name='plasma')
    # display_min = max(1e0, nlp.min())
    # print(display_min)
    norm = LogNorm(1e1, 2e3)
    sm = plt.cm.ScalarMappable(cmap=the_cmap, norm=norm)

    cols_to_use = [col for col in le_data.columns if col != 'neglogpost']
    g = sns.PairGrid(data=le_data, height=10, vars=cols_to_use)
    g.map_lower(sns.scatterplot, alpha=alpha, hue=nlp, hue_norm=norm, palette=the_cmap,
                legend=False)  # alpha=0.002, hue_norm=LogNorm(vmin=nlp.min(), vmax=nlp.max()),
    # g.ax_joint.figure.colorbar(sm, shrink=10.0)
    g.map_diag(sns.histplot, kde=True)
    plt.savefig(f'big_plot_{filename}.svg')
    # plt.show()

create_posterior_big_plot_color(df_results, "filename here")

plt.show()
