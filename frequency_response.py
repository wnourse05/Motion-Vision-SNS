import numpy as np
import matplotlib.pyplot as plt
from sns_toolbox.backends import Backend

def sample_frequency_response(model: Backend, low_hz=0.1, high_hz=100, num_samples=50, plot=True, debug=False):
    dt_s = model.dt/1000
    t_max = (1/low_hz)*3
    t = np.arange(start=0.0, stop=t_max, step=dt_s)

    frequencies = np.logspace(np.log10(low_hz), np.log10(high_hz), num=num_samples)

    inputs = np.zeros([num_samples, len(t)])
    for i in range(num_samples):
        inputs[i,:] = np.sin(2*np.pi*frequencies[i]*t)

    outputs = np.zeros_like(inputs)
    output_peaks = np.zeros_like(frequencies)

    if debug:
        traces = plt.figure()

    for freq in range(num_samples):
        print('Frequency ' + str(freq+1) + '/' + str(num_samples))
        model.reset()
        for step in range(len(t)):
            if t[step] > (3 * (1/frequencies[freq])):
                break
            outputs[freq, step] = model([inputs[freq,step]])
        if debug:
            plt.subplot(num_samples,1,freq+1)
            plt.plot(t[:step], inputs[freq,:step], label='Input')
            plt.plot(t[:step], outputs[freq,:step], label='Output')
        output_peaks[freq] = np.max(outputs[freq,int(step/2):step])
    if debug:
        plt.show()
    # output_peaks = np.max(outputs[:,:(num_samples/2)], axis=1)

    if plot:
        fig = plt.figure()
        plt.plot(frequencies, 20*np.log10(output_peaks), label='Output Response')
        plt.axhline(y=20*np.log10(1/np.sqrt(2)), color='black', label='Cutoff Threshold')
        plt.xlabel('f (Hz)')
        plt.xscale('log')
        plt.ylabel('Magnitude (dB)')
        plt.title('Magnitude Response')
        plt.legend()

        return fig, frequencies, output_peaks
    else:
        return frequencies, output_peaks
