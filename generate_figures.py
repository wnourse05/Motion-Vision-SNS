import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib as mpl
import matplotlib.colors as mc
import seaborn as sea
from utilities import load_data, h5_to_dataframe
import numpy as np
import colorsys

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GLOBAL STYLE
"""
def set_style():
    sea.set_theme(context='notebook', style='darkgrid', palette='colorblind')

def scale_lightness(color, num, max_brightness=1.0):
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
def plot_best_emd_results(best_data, num_best, name, color, a=False, b=True, ratio=False, goal=True, legend=True, filetype=None, log=True, max_brightness=0.9):
    print('Best %i %s Responses'%(num_best, name))
    vels = best_data['vels']
    colors = scale_lightness(color, num_best, max_brightness=max_brightness)

    if a:
        fig_a_peaks = plt.figure()
        plt.title(name+'_a Peak Velocity Response')
        plt.ylabel('Peak Magnitude')
        plt.xlabel('Edge Velocity (deg/s)')
    if b:
        fig_b_peaks = plt.figure()
        plt.title(name+'_b Peak Velocity Response')
        plt.ylabel('Peak Magnitude')
        plt.xlabel('Edge Velocity (deg/s)')
    if ratio:
        fig_ratio = plt.figure()
        plt.title(name+' Velocity Response Ratio')
        plt.ylabel(name+'_b / '+name+'_a')
        plt.xlabel('Edge Velocity (deg/s)')

    for i in range(num_best):
        trial = best_data['trial%i'%i]
        if a:
            plt.figure(fig_a_peaks)
            if i == 0:
                plt.plot(vels, trial['aPeaks'], label='0', lw=2 * mpl.rcParams['lines.linewidth'], zorder=10, color=colors[i])
            else:
                plt.plot(vels, trial['aPeaks'], label='%i'%i, linestyle=':', color=colors[i])
        if b:
            plt.figure(fig_b_peaks)
            if i == 0:
                plt.plot(vels, trial['bPeaks'], label='0', lw=2 * mpl.rcParams['lines.linewidth'], zorder=10, color=colors[i])
            else:
                plt.plot(vels, trial['bPeaks'], label='%i'%i, linestyle=':', color=colors[i])
        if ratio:
            plt.figure(fig_ratio)
            if i == 0:
                plt.plot(vels, trial['ratios'], label='0', lw=2 * mpl.rcParams['lines.linewidth'], zorder=10, color=colors[i])
            else:
                plt.plot(vels, trial['ratios'], label='%i'%i, linestyle=':', color=colors[i])

    if filetype is None:
        filetype = 'svg'
    if a:
        plt.figure(fig_a_peaks)
        if legend:
            plt.legend()
        sea.despine()
        if log:
            plt.xscale('log')
        plt.savefig(f'Figures/{name}_a_peaks_best_{num_best}.{filetype}')
    if b:
        plt.figure(fig_b_peaks)
        if goal:
            plt.plot(vels, best_data['goalResponse'], color='black', linestyle='--', label='Goal')
        if legend:
            plt.legend()
        sea.despine()
        if log:
            plt.xscale('log')
        plt.savefig(f'Figures/{name}_b_peaks_best_{num_best}.{filetype}')
    if ratio:
        plt.figure(fig_ratio)
        if legend:
            plt.legend()
        sea.despine()
        if log:
            plt.xscale('log')
        plt.savefig(f'Figures/{name}_ratios_best_{num_best}.{filetype}')

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Best EMD Parameters
"""
def setup_number_line(ax, title, unit, data):
    # only show the bottom spine
    ax.yaxis.set_major_locator(ticker.NullLocator())
    # ax.spines[['left', 'right', 'top']].set_visible(False)

    # define tick positions
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(1.00))
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))

    # ax.xaxis.set_ticks_position('bottom')
    # ax.tick_params(which='major', width=1.00, length=5)
    # ax.tick_params(which='minor', width=0.75, length=2.5, labelsize=10)
    # ax.text(0.0, 0.2, title, transform=ax.transAxes, color='black')
    ax.xaxis.set_major_formatter('{x:.2f} %s'%unit)
    ax.set_title(title)
    # ax.hlines(y=0, xmin=min(data), xmax=max(data), colors='black')


def plot_best_emd_parameters(best_data, num_best, name, param_units, filetype=None):
    print('Best %i %s Parameters' % (num_best, name))
    params = list(best_data['best'].columns.values)
    num_params = len(params)
    fig, ax = plt.subplots(num_params-1, 1)
    fig.suptitle('%s Best Parameter Values'%name)
    for i in range(1, num_params):
        for j in range(num_best):
            # plt.subplot(num_params-1, 1, i)
            setup_number_line(ax[i-1], params[i], param_units[i-1], best_data['best'][params[i]])
            markersize = mpl.rcParams['lines.markersize']**2
            if j == 0:
                ax[i-1].scatter(best_data['best'][params[i]].iloc[j], 0, marker='*', s=4*markersize, zorder=10)
            else:
                ax[i-1].scatter(best_data['best'][params[i]].iloc[j], 0, s=markersize)
    if filetype is None:
        filetype = 'svg'
    plt.tight_layout()
    plt.savefig(f'Figures/{name}_params_best_{num_best}.{filetype}')

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Landscape Correlogram (MCMC Results)
"""
def mcmc_correlogram(h5, toml, params_used, name, style=None, cmap=None, filetype=None, alpha=0.05):
    print('%s MCMC Results'%name)
    df_results = h5_to_dataframe(h5, toml, params_used)
    df_results['neglogpost'] = df_results['neglogpost'] - np.min(df_results['neglogpost']) + 1
    df_results = df_results.sort_values(by='neglogpost', ignore_index=True, ascending=False)

    if style is None:
        style = 'white'
    if cmap is None:
        cmap = 'magma'
    the_cmap = sea.color_palette(cmap, as_cmap=True)
    norm = mc.LogNorm(np.min(df_results['neglogpost']), np.max(df_results['neglogpost']))
    sea.set_context('paper', rc={"font.size": 48, "axes.titlesize": 48, "axes.labelsize": 48, "xtick.labelsize": 48,
                                 "ytick.labelsize": 48})
    sea.set_style(style)
    le_data = df_results.sort_values(by=['neglogpost'], ignore_index=True, ascending=False)
    sample_size = int(1.0 * len(le_data['neglogpost']))
    nlp = (le_data['neglogpost'])  # .sample(n=sample_size)

    cols_to_use = [col for col in le_data.columns if col != 'neglogpost']
    g = sea.PairGrid(data=le_data, height=10, vars=cols_to_use)
    g.map_offdiag(sea.scatterplot, alpha=alpha, hue=nlp, hue_norm=norm, palette=the_cmap,
                  legend=False)
    # g.map_lower(sns.scatterplot, alpha=alpha, hue=nlp, hue_norm=norm, palette=the_cmap,
    #             legend=False)  # alpha=0.002, hue_norm=LogNorm(vmin=nlp.min(), vmax=nlp.max()),
    # g.map_upper(sns.kdeplot, fill=True, thresh=0)#, hue=nlp, hue_norm=norm, palette=the_cmap)
    # g.ax_joint.figure.colorbar(sm, shrink=10.0)
    g.map_diag(sea.histplot, kde=True)

    if filetype is None:
        filetype = 'png'
    plt.savefig(f'Figures/{name}_mcmc.{filetype}')

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
Run Everything
"""
def main():
    # colors from Paul Tol
    colors = {'retina': '#CC79A7',  # purple
              'l1':     '#44AA99',  # teal
              'l2':     '#44AA99',
              'l3':     '#DDCC77',  # sand
              'mi1':    '#88CCEE',  # cyan
              'tm1':    '#88CCEE',
              'mi9':    '#117733',  # green
              'tm9':    '#117733',
              'ct1on':  '#CC6677',  # rose
              'ct1off': '#CC6677',
              't4':     '#332288',  # indigo
              't5':     '#882255'}  # wine
    set_style()
    t4_best = load_data('t4_best_results.pc')
    t5_best = load_data('t5_best_results.pc')

    # plot_best_emd_results(t4_best, 5, 'T4', color=colors['t4'])
    # plot_best_emd_results(t4_best, 10, 'T4', color=colors['t4'])
    # plot_best_emd_results(t5_best, 5, 'T5', color=colors['t5'])
    # plot_best_emd_results(t5_best, 10, 'T5', color=colors['t5'])

    # plot_best_emd_parameters(t4_best, 5, 'T4', ['Hz', '', 'Hz', 'Hz', ''])
    # plot_best_emd_parameters(t4_best, 10, 'T4', ['Hz', '', 'Hz', 'Hz', ''])
    # plot_best_emd_parameters(t5_best, 5, 'T5', ['Hz', '', 'Hz', 'uS', 'mV'])
    # plot_best_emd_parameters(t5_best, 10, 'T5', ['Hz', '', 'Hz', 'uS', 'mV'])

    dim = 7*4
    fov_res = 5
    fov = fov_res*dim
    wavelength = fov/4
    angles = np.arange(0,360,45)
    plot_sinusoids(wavelength, angles, fov, fov_res, title=False)

    # params_used = np.array([0,1,2,3,4])
    # t4_h5 = '2023-Mar-24_07-15.t4a.h5'
    # t4_toml = "conf_t4_reduced.toml"
    # t5_h5 = '2023-Mar-25_05-40_t5.h5'
    # t5_toml = 'conf_t5_mcmc.toml'
    # mcmc_correlogram(t4_h5, t4_toml, params_used, 'T4')
    # mcmc_correlogram(t5_h5, t5_toml, params_used, 'T5')

    plt.show()

if __name__ == '__main__':
    main()
