import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def gen_plot(data, gains, freqs, title, relative=True):
    plt.figure()
    index = 0
    if not relative:
        upper_bound = np.max(data)
        norm = plt.Normalize(-1.5, 1.5)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['blue', "white", 'red'])
    for c in range(len(gains)):
        for l in range(len(gains)):
            index += 1
            if len(freqs)>1:
                graph = data[l,c,:,:]
            else:
                graph = data[l,c,:]
                graph = np.expand_dims(graph, axis=1)
            plt.subplot(len(gains), len(gains), index)
            if not relative:
                plt.imshow(graph, interpolation='none', norm=norm, cmap=cmap)
            else:
                plt.imshow(graph, interpolation='none')
            plt.ylabel('center: %f'%(gains[c]))
            plt.title('left: %f'%(gains[l]))
            plt.colorbar()
            ax = plt.gca()

            ax.set_xticks(np.arange(0, len(freqs), 1))
            ax.set_xticks(np.arange(-0.5, len(freqs), 1), minor=True)
            ax.set_xticklabels(freqs.round(decimals=1))
            ax.set_yticks(np.arange(0, len(gains), 1))
            ax.set_yticks(np.arange(-0.5, len(gains), 1), minor=True)
            ax.set_yticklabels(gains)
            ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
            ax.tick_params(which='minor', bottom=False, left=False)
            # for r in range(len(gains)):
            #     row = data_sns_toolbox[l,c,r,:]
    plt.suptitle(title)


data = pickle.load(open('emd_freq_gain_data.p', 'rb'))
gains = data['gains']
freqs = data['frequencies']
peaks_t4 = data['peaksT4']
peaks_t5 = data['peaksT5']
ratios_t4 = data['ratiosT4']
ratios_t5 = data['ratiosT5']

gen_plot(peaks_t4, gains, freqs, 'T4 (Relative)', relative=True)
gen_plot(peaks_t4, gains, freqs, 'T4', relative=False)

gen_plot(peaks_t5, gains, freqs, 'T5 (Relative)', relative=True)
gen_plot(peaks_t5, gains, freqs, 'T5', relative=False)

maxvar = peaks_t4.max(axis=3)
minvar = peaks_t4.min(axis=3)
diffvar = maxvar-minvar
index = np.unravel_index(np.argmax(diffvar), diffvar.shape)
print(index)

maxratio = ratios_t4.max(axis=3)
index_ratio = np.unravel_index(np.argmax(maxratio), maxratio.shape)
print(index_ratio)

gen_plot(diffvar, gains, np.array([0]), 'Diff', relative=True)

gen_plot(ratios_t4, gains, freqs, 'T4 Ratio (Relative)', relative=True)
gen_plot(ratios_t4, gains, freqs, 'T4 Ratio', relative=False)
#
gen_plot(ratios_t5, gains, freqs, 'T5 Ratio (Relative)', relative=True)
gen_plot(ratios_t5, gains, freqs, 'T5 Ratio', relative=False)

plt.show()
