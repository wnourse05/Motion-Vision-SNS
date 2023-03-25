import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib as mpl
import seaborn as sea
from utilities import load_data

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
GLOBAL STYLE
"""
def set_style():
    sea.set_theme(context='notebook', style='ticks', palette='colorblind')

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Best EMD Results
"""
def plot_best_emd_results(best_data, num_best, name, a=True, b=True, ratio=True, goal=True, legend=True, filetype=None, log=True):
    print('Best %i %s Responses'%(num_best, name))
    vels = best_data['vels']

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
        plt.title(name+'_a Velocity Response Ratio')
        plt.ylabel(name+'_a / '+name+'_b')
        plt.xlabel('Edge Velocity (deg/s)')

    for i in range(num_best):
        trial = best_data['trial%i'%i]
        if a:
            plt.figure(fig_a_peaks)
            if i == 0:
                plt.plot(vels, trial['aPeaks'], label='0', lw=2 * mpl.rcParams['lines.linewidth'], zorder=10)
            else:
                plt.plot(vels, trial['aPeaks'], label='%i'%i, linestyle=':')
        if b:
            plt.figure(fig_b_peaks)
            if i == 0:
                plt.plot(vels, trial['bPeaks'], label='0', lw=2 * mpl.rcParams['lines.linewidth'], zorder=10)
            else:
                plt.plot(vels, trial['bPeaks'], label='%i'%i, linestyle=':')
        if ratio:
            plt.figure(fig_ratio)
            if i == 0:
                plt.plot(vels, trial['ratios'], label='0', lw=2 * mpl.rcParams['lines.linewidth'], zorder=10)
            else:
                plt.plot(vels, trial['ratios'], label='%i'%i, linestyle=':')

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
Run Everything
"""
def main():
    set_style()
    t4_best = load_data('t4_best_results.pc')

    plot_best_emd_results(t4_best, 5, 'T4')
    plot_best_emd_results(t4_best, 10, 'T4')

    plot_best_emd_parameters(t4_best, 5, 'T4', ['Hz', '', 'Hz', 'Hz', ''])
    plot_best_emd_parameters(t4_best, 10, 'T4', ['Hz', '', 'Hz', 'Hz', ''])

    plt.show()

if __name__ == '__main__':
    main()
