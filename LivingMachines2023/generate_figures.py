import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib as mpl
import matplotlib.colors as mc
import seaborn as sea
from utilities import load_data, h5_to_dataframe
import numpy as np
import colorsys
import pandas as pd

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GLOBAL STYLE
"""
def set_style():
    sea.set_theme(context='notebook', style='ticks', palette='colorblind')

def scale_lightness(color, num, max_brightness=0.9):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    current = c[1]
    lightness = np.geomspace(current, max_brightness, num)
    colors = []
    for i in range(num):
        colors.append(colorsys.hls_to_rgb(c[0], lightness[i], c[2]))
    return colors

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Best EMD Results
"""
# def plot_best_emd_results(best_data, num_best, name, color, a=False, b=True, ratio=True, goal=True, legend=True, filetype=None, log=True, max_brightness=0.9):
#     print('Best %i %s Responses'%(num_best, name))
#     vels = best_data['vels']
#     colors = scale_lightness(color, num_best, max_brightness=max_brightness)
#
#     if a:
#         fig_a_peaks = plt.figure()
#         plt.title(name+'_a Peak Velocity Response')
#         plt.ylabel('Peak Magnitude')
#         plt.xlabel('Edge Velocity (deg/s)')
#     if b:
#         fig_b_peaks = plt.figure()
#         plt.title(name+'_b Peak Velocity Response')
#         plt.ylabel('Peak Magnitude')
#         plt.xlabel('Edge Velocity (deg/s)')
#     if ratio:
#         fig_ratio = plt.figure()
#         plt.title(name+' Velocity Response Ratio')
#         plt.ylabel(name+'_b / '+name+'_a')
#         plt.xlabel('Edge Velocity (deg/s)')
#
#     for i in range(num_best):
#         trial = best_data['trial%i'%i]
#         if a:
#             plt.figure(fig_a_peaks)
#             if i == 0:
#                 plt.plot(vels, trial['aPeaks'], label='0', lw=2 * mpl.rcParams['lines.linewidth'], zorder=10, color=colors[i])
#             else:
#                 plt.plot(vels, trial['aPeaks'], label='%i'%i, linestyle=':', color=colors[i])
#         if b:
#             plt.figure(fig_b_peaks)
#             if i == 0:
#                 plt.plot(vels, trial['bPeaks'], label='0', lw=2 * mpl.rcParams['lines.linewidth'], zorder=10, color=colors[i])
#             else:
#                 plt.plot(vels, trial['bPeaks'], label='%i'%i, linestyle=':', color=colors[i])
#         if ratio:
#             plt.figure(fig_ratio)
#             if i == 0:
#                 plt.plot(vels, trial['ratios'], label='0', lw=2 * mpl.rcParams['lines.linewidth'], zorder=10, color=colors[i])
#             else:
#                 plt.plot(vels, trial['ratios'], label='%i'%i, linestyle=':', color=colors[i])
#
#     if filetype is None:
#         filetype = 'svg'
#     if a:
#         plt.figure(fig_a_peaks)
#         if legend:
#             plt.legend()
#         sea.despine()
#         if log:
#             plt.xscale('log')
#         plt.savefig(f'Figures/{name}_a_peaks_best_{num_best}.{filetype}')
#     if b:
#         plt.figure(fig_b_peaks)
#         if goal:
#             plt.plot(vels, best_data['goalResponse'], color='black', linestyle='--', label='Goal')
#         if legend:
#             plt.legend()
#         sea.despine()
#         if log:
#             plt.xscale('log')
#         plt.savefig(f'Figures/{name}_b_peaks_best_{num_best}.{filetype}')
#     if ratio:
#         plt.figure(fig_ratio)
#         if legend:
#             plt.legend()
#         sea.despine()
#         if log:
#             plt.xscale('log')
#         plt.savefig(f'Figures/{name}_ratios_best_{num_best}.{filetype}')
#
# """
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Best EMD Parameters
# """
# def setup_number_line(ax, title, unit, data_sns_toolbox):
#     # only show the bottom spine
#     ax.yaxis.set_major_locator(ticker.NullLocator())
#     # ax.spines[['left', 'right', 'top']].set_visible(False)
#
#     # define tick positions
#     # ax.xaxis.set_major_locator(ticker.MultipleLocator(1.00))
#     # ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
#
#     # ax.xaxis.set_ticks_position('bottom')
#     # ax.tick_params(which='major', width=1.00, length=5)
#     # ax.tick_params(which='minor', width=0.75, length=2.5, labelsize=10)
#     # ax.text(0.0, 0.2, title, transform=ax.transAxes, color='black')
#     ax.xaxis.set_major_formatter('{x:.2f} %s'%unit)
#     ax.set_title(title)
#     # ax.hlines(y=0, xmin=min(data_sns_toolbox), xmax=max(data_sns_toolbox), colors='black')
#
#
# def plot_best_emd_parameters(best_data, num_best, name, param_units, filetype=None):
#     print('Best %i %s Parameters' % (num_best, name))
#     params = list(best_data['best'].columns.values)
#     num_params = len(params)
#     fig, ax = plt.subplots(num_params-1, 1)
#     fig.suptitle('%s Best Parameter Values'%name)
#     for i in range(1, num_params):
#         for j in range(num_best):
#             # plt.subplot(num_params-1, 1, i)
#             setup_number_line(ax[i-1], params[i], param_units[i-1], best_data['best'][params[i]])
#             markersize = mpl.rcParams['lines.markersize']**2
#             if j == 0:
#                 ax[i-1].scatter(best_data['best'][params[i]].iloc[j], 0, marker='*', s=4*markersize, zorder=10)
#             else:
#                 ax[i-1].scatter(best_data['best'][params[i]].iloc[j], 0, s=markersize)
#     if filetype is None:
#         filetype = 'svg'
#     plt.tight_layout()
#     plt.savefig(f'Figures/{name}_params_best_{num_best}.{filetype}')
#
# """
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Landscape Correlogram (MCMC Results)
# """
# def mcmc_correlogram(h5, toml, params_used, name, style=None, cmap=None, filetype=None, alpha=0.05):
#     print('%s MCMC Results'%name)
#     df_results = h5_to_dataframe(h5, toml, params_used)
#     df_results['neglogpost'] = df_results['neglogpost'] - np.min(df_results['neglogpost']) + 1
#     df_results = df_results.sort_values(by='neglogpost', ignore_index=True, ascending=False)
#
#     if style is None:
#         style = 'white'
#     if cmap is None:
#         cmap = 'magma'
#     the_cmap = sea.color_palette(cmap, as_cmap=True)
#     norm = mc.LogNorm(np.min(df_results['neglogpost']), np.max(df_results['neglogpost']))
#
#     fig, ax = plt.subplots(figsize=(6, 1))
#     fig.subplots_adjust(bottom=0.5)
#     fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=the_cmap),
#                  cax=ax, orientation='horizontal')
#     plt.tick_params(axis='x', which='minor', bottom=False)
#     plt.savefig(f'Figures/{name}_mcmc_colorbar.svg')
#
#     sea.set_context('paper', rc={"font.size": 48, "axes.titlesize": 48, "axes.labelsize": 48, "xtick.labelsize": 48,
#                                  "ytick.labelsize": 48})
#     sea.set_style(style)
#     le_data = df_results.sort_values(by=['neglogpost'], ignore_index=True, ascending=False)
#     sample_size = int(1.0 * len(le_data['neglogpost']))
#     nlp = (le_data['neglogpost'])  # .sample(n=sample_size)
#
#     cols_to_use = [col for col in le_data.columns if col != 'neglogpost']
#     g = sea.PairGrid(data_sns_toolbox=le_data, height=10, vars=cols_to_use)
#     g.map_offdiag(sea.scatterplot, alpha=alpha, hue=nlp, hue_norm=norm, palette=the_cmap,
#                   legend=False)
#     g.map_diag(sea.histplot, kde=True)
#
#     best = le_data.iloc[-1]
#     best_df = pd.DataFrame(np.reshape(best.values, (1, 6)), columns=best.index.values)
#     blue = sea.color_palette('colorblind')[0]
#     markersize = mpl.rcParams['lines.markersize'] ** 2
#     cols_to_use = [col for col in best_df.columns if col != 'neglogpost']
#     g.data_sns_toolbox = best_df
#     g.map_offdiag(sea.scatterplot, alpha=1, color=blue, marker='*', s=100*markersize, legend=False)
#
#     if filetype is None:
#         filetype = 'png'
#     plt.savefig(f'Figures/{name}_mcmc.{filetype}')



"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Angled Sinusoids
"""
def plot_sinusoids(wavelength, angles, fov, res, title=True, filetype=None):
    for i in range(len(angles)):
        plt.figure()
        x = np.arange(0, fov, res)
        x_rad = np.deg2rad(x)
        X, Y = np.meshgrid(x_rad, x_rad)
        Y = np.flipud(Y)
        wavelength_rad = np.deg2rad(wavelength)
        angle_rad = np.deg2rad(angles[i])
        grating = 0.5 * np.sin(2 * np.pi * (X * np.cos(angle_rad) + Y * np.sin(angle_rad)) / wavelength_rad) + 0.5
        grating[grating > 0.5] = 1.0
        grating[grating <= 0.5] = 0.0
        if title:
            plt.title('%i' % (angles[i]))
        if filetype is None:
            filetype = 'svg'
        cmap = 'gray'
        plt.set_cmap(cmap)
        plt.imshow(grating, interpolation='none')
        plt.axis('off')

        plt.savefig('Figures/grating_%i.%s'%(angles[i],filetype))

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Radar Plots
"""
def radar(angles, key, title, color, legend=False):
    # plt.figure()
    # ax = plt.subplot(111, polar=True)
    angles = np.deg2rad(angles)
    angles = [n for n in angles]
    # angles.append(0)

    peaks = np.zeros(len(angles))
    for j in range(len(angles)-1):
        data = load_data('Radar Data/peaks_%i_%i.pc'%(0,j))
        peaks[j] = data['%s'%key].item()
    peaks[-1] = peaks[0]
    # print(peaks)

    plt.plot(angles, peaks, color=color)
    yticksize = mpl.rcParams['ytick.labelsize']
    plt.yticks(fontsize=yticksize*0.75)

    plt.title(title)
    if legend:
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, mode="expand", borderaxespad=0.)

def total_radar(colors):
    sea.set(font_scale=2)
    angles = [0, 45, 90, 135, 180, 225, 270, 315, 0]
    plt.figure(figsize=(2*7, 2*7*9/16))
    plt.subplot(2, 4, 1, polar=True)
    radar(angles, 'on_a', r'On$_A$', colors['on'])
    plt.subplot(2, 4, 2, polar=True)
    radar(angles, 'on_b', r'On$_B$', colors['on'])
    plt.subplot(2, 4, 3, polar=True)
    radar(angles, 'off_a', r'Off$_A$', colors['off'])
    plt.subplot(2, 4, 4, polar=True)
    radar(angles, 'off_b', r'Off$_B$', colors['off'])
    plt.subplot(2, 4, 5, polar=True)
    radar(angles, 'on_c', r'On$_C$', colors['on'])
    plt.subplot(2, 4, 6, polar=True)
    radar(angles, 'on_d', r'On$_D$', colors['on'])
    plt.subplot(2, 4, 7, polar=True)
    radar(angles, 'off_c', r'Off$_C$', colors['off'])
    plt.subplot(2, 4, 8, polar=True)
    radar(angles, 'off_d', r'Off$_D$', colors['off'])
    plt.tight_layout()
    plt.savefig('Figures/figure_radar.svg')
    plt.savefig('Figures/figure_radar.pdf')

# def resultant_radar(colors)

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Step Responses
"""
def step_on(colors, title=True):
    data = load_data('Step Responses/positive.pc')
    if title:
        plt.title('On Pathway')
    plt.plot(data['t'][:int(len(data['t'])/2)], data['in'][:int(len(data['t'])/2)], color=colors['in'], label=r'$In$')
    plt.plot(data['t'][:int(len(data['t'])/2)], data['bp'][:int(len(data['t'])/2)], color=colors['bp'], label=r'$B_O$')
    plt.plot(data['t'][:int(len(data['t'])/2)], data['lp'][:int(len(data['t'])/2)], color=colors['lp'], label=r'$L$')
    plt.plot(data['t'][:int(len(data['t'])/2)], data['e'][:int(len(data['t'])/2)], color=colors['e'], label=r'$E_O$')
    plt.plot(data['t'][:int(len(data['t'])/2)], data['d_on'][:int(len(data['t'])/2)], color=colors['d'], label=r'$D_O$')
    plt.plot(data['t'][:int(len(data['t'])/2)], data['s_on'][:int(len(data['t'])/2)], color=colors['s'], label=r'$S_O$')
    plt.legend(loc=4)
    plt.xlabel('t (ms)')

def step_off(colors, title=True):
    data = load_data('Step Responses/negative.pc')
    if title:
        plt.title('Off Pathway')
    plt.plot(data['t'][int(len(data['t'])/2):], data['in'][int(len(data['t'])/2):], color=colors['in'], label=r'$In$')
    plt.plot(data['t'][int(len(data['t'])/2):], data['bp'][int(len(data['t'])/2):], color=colors['bp'], label=r'$B_F$')
    plt.plot(data['t'][int(len(data['t'])/2):], data['lp'][int(len(data['t'])/2):], color=colors['lp'], label=r'$L$')
    plt.plot(data['t'][int(len(data['t'])/2):], data['e'][int(len(data['t'])/2):], color=colors['e'], label=r'$E_F$')
    plt.plot(data['t'][int(len(data['t'])/2):], data['d_off'][int(len(data['t'])/2):], color=colors['d'], label=r'$D_F$')
    plt.plot(data['t'][int(len(data['t'])/2):], data['s_off'][int(len(data['t'])/2):], color=colors['s'], label=r'$S_F$')
    plt.legend()
    plt.xlabel('t (ms)')

def step_figure(colors, title=True):
    plt.figure(figsize=(7, 7*9/16))
    plt.subplot(1,2,1)
    step_on(colors, title=title)
    plt.subplot(1,2,2)
    step_off(colors, title=title)
    plt.tight_layout()
    plt.savefig('Figures/figure_step.svg')
    plt.savefig('Figures/figure_step.pdf')


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
EMD Operation
"""
def plot_emd(index, colors, on=True):
    stim = load_data('Neuron Data/stim_0_%i.pc'%index)
    emd = load_data('Neuron Data/emd_0_%i.pc'%index)
    emd_in = load_data('Neuron Data/emd_in_0_%i.pc'%index)
    t = stim['t']
    stim_example = stim['stim']

    plt.subplot(3,1,1)
    plt.plot(t, stim_example)
    plt.subplot(3,1,2)
    if on:
        plt.plot(t, emd_in['e'], color=colors['e'], label='E On')
        plt.plot(t, emd_in['d_on'], color=colors['d'], label='D On')
        plt.plot(t, emd_in['s_on'], color=colors['s'], label='S On')
    else:
        plt.plot(t, emd_in['e'], color=colors['e'], label='E Off')
        plt.plot(t, emd_in['d_off'], color=colors['d'], label='D Off')
        plt.plot(t, emd_in['s_off'], color=colors['s'], label='S Off')
    plt.legend()
    plt.subplot(3,1,3)
    if on:
        plt.plot(t, emd['on_b'], color=colors['on'])
    else:
        plt.plot(t, emd['off_b'], color=colors['off'])

def plot_on(colors):
    index = 0
    stim_lr = load_data('Neuron Data/stim_0_%i.pc' % index)
    emd_lr = load_data('Neuron Data/emd_0_%i.pc' % index)
    emd_in_lr = load_data('Neuron Data/emd_in_0_%i.pc' % index)
    t = stim_lr['t']
    stim_example_lr = stim_lr['stim']
    window_lr = [990, 1490]
    window_y = [-0.5,0.5]
    # window_lr = [0,4000]

    fig = plt.figure(figsize=(7, 7*9/16))

    plt.subplot(3,2,1)
    plt.title('Preferred Direction')
    plt.plot(t, stim_example_lr[:,0], color=colors['in'], linestyle='--', label='In (Left)')
    plt.plot(t, stim_example_lr[:,1], color=colors['in'], label='In (Center)')
    plt.plot(t, stim_example_lr[:,2], color=colors['in'], linestyle=':', label='In (Right)')
    # plt.legend()
    plt.xlim(window_lr)
    plt.subplot(3,2,3)
    plt.plot(t, emd_in_lr['e'], color=colors['e'], linestyle='--', label=r'$E_O$')
    plt.plot(t, emd_in_lr['d_on'], color=colors['d'], label=r'$D_O$')
    plt.plot(t, emd_in_lr['s_on'], color=colors['s'], linestyle=':', label=r'$S_O$')
    # plt.legend()
    plt.xlim(window_lr)
    plt.subplot(3,2,5)
    plt.plot(t, emd_lr['on_b'], color=colors['on'], label=r'$On_B$')
    # plt.legend()
    plt.xlim(window_lr)
    plt.ylim(window_y)
    plt.xlabel('t (ms)')

    index = 1
    stim_rl = load_data('Neuron Data/stim_0_%i.pc' % index)
    emd_rl = load_data('Neuron Data/emd_0_%i.pc' % index)
    emd_in_rl = load_data('Neuron Data/emd_in_0_%i.pc' % index)
    t = stim_rl['t']
    stim_example_rl = stim_rl['stim']
    window_rl = [0, 600]
    # window_rl = [0, 4000]

    plt.subplot(3, 2, 2)
    plt.title('Null Direction')
    plt.plot(t, stim_example_rl[:,0], color=colors['in'], linestyle='--')
    plt.plot(t, stim_example_rl[:,1], color=colors['in'])
    plt.plot(t, stim_example_rl[:,2], color=colors['in'], linestyle=':')
    # plt.legend()
    plt.xlim(window_rl)
    plt.subplot(3, 2, 4)
    plt.plot(t, emd_in_rl['e'], color=colors['e'], linestyle='--')
    plt.plot(t, emd_in_rl['d_on'], color=colors['d'])
    plt.plot(t, emd_in_rl['s_on'], color=colors['s'], linestyle=':')
    # plt.legend()
    plt.xlim(window_rl)
    plt.subplot(3, 2, 6)
    plt.plot(t, emd_rl['on_b'], color=colors['on'])
    # plt.legend()
    plt.xlim(window_rl)
    plt.ylim(window_y)
    plt.xlabel('t (ms)')


    plt.tight_layout()
    fig.legend(loc=8, mode='expand', ncol=7)  # , bbox_to_anchor=(0.5, 0.0))

    plt.savefig('Figures/figure_on.svg')
    plt.savefig('Figures/figure_on.pdf')

def plot_off(colors):
    index = 0
    stim_lr = load_data('Neuron Data/stim_0_%i.pc' % index)
    emd_lr = load_data('Neuron Data/emd_0_%i.pc' % index)
    emd_in_lr = load_data('Neuron Data/emd_in_0_%i.pc' % index)
    t = stim_lr['t']
    stim_example_lr = stim_lr['stim']
    window_lr = [490, 990]
    window_y = [-1,1]
    # window_lr = [0, 4000]

    fig = plt.figure(figsize=(7, 7 * 9 / 16))

    plt.subplot(3, 2, 1)
    plt.title('Preferred Direction')
    plt.plot(t, stim_example_lr[:,0], color=colors['in'], linestyle='--', label='In (Left)')
    plt.plot(t, stim_example_lr[:,1], color=colors['in'], label='In (Center)')
    plt.plot(t, stim_example_lr[:,2], color=colors['in'], linestyle=':', label='In (Right)')
    # plt.legend()
    plt.xlim(window_lr)
    plt.subplot(3, 2, 3)
    plt.plot(t, emd_in_lr['e'], color=colors['e'], linestyle='--', label=r'$E_F$')
    plt.plot(t, emd_in_lr['d_off'], color=colors['d'], label=r'$D_F$')
    plt.plot(t, emd_in_lr['s_off'], color=colors['s'], linestyle=':', label=r'$S_F$')
    # plt.legend()
    plt.xlim(window_lr)
    plt.subplot(3, 2, 5)
    plt.plot(t, emd_lr['off_b'], color=colors['off'], label=r'$Off_B$')
    # plt.legend()
    plt.xlim(window_lr)
    plt.ylim(window_y)
    plt.xlabel('t (ms)')

    index = 1
    stim_rl = load_data('Neuron Data/stim_0_%i.pc' % index)
    emd_rl = load_data('Neuron Data/emd_0_%i.pc' % index)
    emd_in_rl = load_data('Neuron Data/emd_in_0_%i.pc' % index)
    t = stim_rl['t']
    stim_example_rl = stim_rl['stim']
    window_rl = [490, 990]
    # window_rl = [0,4000]

    plt.subplot(3, 2, 2)
    plt.title('Null Direction')
    plt.plot(t, stim_example_rl[:,0], color=colors['in'], linestyle='--')
    plt.plot(t, stim_example_rl[:,1], color=colors['in'])
    plt.plot(t, stim_example_rl[:,2], color=colors['in'], linestyle=':')
    # plt.legend()
    plt.xlim(window_rl)
    plt.subplot(3, 2, 4)
    plt.plot(t, emd_in_rl['e'], color=colors['e'], linestyle='--')
    plt.plot(t, emd_in_rl['d_off'], color=colors['d'])
    plt.plot(t, emd_in_rl['s_off'], color=colors['s'], linestyle=':')
    # plt.legend()
    plt.xlim(window_rl)
    plt.subplot(3, 2, 6)
    plt.plot(t, emd_rl['off_b'], color=colors['off'])
    # plt.legend()
    plt.xlim(window_rl)
    plt.ylim(window_y)
    plt.xlabel('t (ms)')


    plt.tight_layout()
    fig.legend(loc=8, mode='expand', ncol=7)  # , bbox_to_anchor=(0.5, 0.0))


    plt.savefig('Figures/figure_off.svg')
    plt.savefig('Figures/figure_off.pdf')

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Frequency Response
"""
def frequency_response(vels, angle=0, on=True):
    peaks = np.zeros(len(vels))
    ratios = np.zeros(len(vels))
    for i in range(len(vels)):
        data = load_data('Frequency Response/peaks_%i_%i.pc'%(i, angle))
        if on:
            peaks[i] = data['on_b']
            ratios[i] = data['on_b']/data['on_a']
        else:
            peaks[i] = data['off_b']
            ratios[i] = data['off_b'] / data['off_a']
    plt.subplot(2,1,1)
    plt.plot(vels, peaks)
    plt.xscale('log')
    plt.subplot(2,1,2)
    plt.plot(vels, ratios)
    plt.xscale('log')

def plot_frequency_response(vels, colors):
    peaks_on = np.zeros(len(vels))
    ratios_on = np.zeros(len(vels))
    peaks_off = np.zeros(len(vels))
    ratios_off = np.zeros(len(vels))
    for i in range(len(vels)):
        data = load_data('Frequency Response/peaks_%i_%i.pc' % (i, 0))
        peaks_on[i] = data['on_b']
        ratios_on[i] = data['on_b'] / data['on_a']
        peaks_off[i] = data['off_b']
        ratios_off[i] = data['off_b'] / data['off_a']

    plt.figure(figsize=(7, 7 * 9 / 16))
    plt.subplot(2,1,1)
    plt.plot(vels, peaks_on, color=colors['on'])
    plt.plot(vels, peaks_off, color=colors['off'], linestyle='--')
    plt.axvline(x=180,linestyle='--',color='black')
    plt.xscale('log')
    plt.ylabel('Peak Magnitude')
    plt.ylim([0,None])
    # plt.title('On Pathway')
    plt.subplot(2,1,2)
    plt.plot(vels, ratios_on, color=colors['on'])
    plt.plot(vels, ratios_off, color=colors['off'], linestyle='--')
    plt.axvline(x=180,linestyle='--',color='black')
    # plt.axhline(y=1,linestyle='--',color='black')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'Velocity ($^{\circ}$/s)')
    plt.ylabel('PD/ND')
    plt.ylim([1, None])
    # plt.figure(figsize=(7, 7 * 9 / 16))
    # plt.subplot(2,2,1)
    # plt.plot(vels, peaks_on, color=colors['on'])
    # plt.axvline(x=180,linestyle='--',color='black')
    # plt.xscale('log')
    # plt.ylabel('Peak Magnitude')
    # plt.title('On Pathway')
    # plt.subplot(2,2,3)
    # plt.plot(vels, ratios_on, color=colors['on'])
    # plt.axvline(x=180,linestyle='--',color='black')
    # plt.xscale('log')
    # plt.xlabel('Velocity (deg/s)')
    # plt.ylabel('PD/ND')
    #
    # plt.subplot(2, 2, 2)
    # plt.plot(vels, peaks_off, color=colors['off'])
    # plt.axvline(x=180,linestyle='--',color='black')
    # plt.xscale('log')
    # # plt.ylabel('Peak Magnitude')
    # plt.title('Off Pathway')
    # plt.subplot(2, 2, 4)
    # plt.plot(vels, ratios_off, color=colors['off'])
    # plt.axvline(x=180,linestyle='--',color='black')
    # plt.xscale('log')
    # plt.xlabel('Velocity (deg/s)')
    # # plt.ylabel('PD/ND')
    plt.tight_layout()
    plt.savefig('Figures/figure_frequency.svg')
    plt.savefig('Figures/figure_frequency.pdf')

def bp_traces(colors):
    saved = load_data('bp_step_response.pc')

    t = saved['t']
    inp = saved['inp']
    data = saved['data_sns_toolbox']

    plt.figure()
    plt.plot(t, inp, color=colors['in'], label='Input')
    plt.plot(t, data[0, :], color='C0', linestyle='--', label='I')
    plt.plot(t, data[1, :], color='C3', linestyle=':', label='Fast')
    plt.plot(t, data[2, :], color='C7', linestyle='dashdot', label='Slow')
    plt.plot(t, data[3, :], color=colors['bp'], label='Out')
    plt.ylabel('Response')
    plt.xlabel('t (ms)')
    plt.legend()
    plt.tight_layout()


    plt.savefig('Figures/figure_bp.svg')
    plt.savefig('Figures/figure_bp.pdf')

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Run Everything
"""
def main():
    # colors from Paul Tol
    # colors = {'retina': '#CC79A7',  # purple
    #           'l1':     '#44AA99',  # teal
    #           'l2':     '#44AA99',
    #           'l3':     '#DDCC77',  # sand
    #           'mi1':    '#88CCEE',  # cyan
    #           'tm1':    '#88CCEE',
    #           'mi9':    '#117733',  # green
    #           'tm9':    '#117733',
    #           'ct1on':  '#CC6677',  # rose
    #           'ct1off': '#CC6677',
    #           't4':     '#332288',  # indigo
    #           't5':     '#882255'}  # wine
    colors = {'in': '#CC79A7',  # purple
              'bp': '#44AA99',  # teal
              'lp': '#DDCC77',  # sand
              'd': '#88CCEE',  # cyan
              'e': '#117733',  # green
              's': '#CC6677',  # rose
              'on': '#332288',  # indigo
              'off': '#999933'}  # olive
    set_style()

    # step_figure(colors)

    # plot_on(colors)
    # plot_off(colors)

    # vels = np.geomspace(10, 360, 10)
    # plot_frequency_response(vels, colors)

    # total_radar(colors)

    bp_traces(colors)

    plt.show()

if __name__ == '__main__':
    main()
