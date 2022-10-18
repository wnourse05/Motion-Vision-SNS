import pickle

from frequency_response import sample_frequency_response

from neurons_and_networks import filter_lowpass

dt = 0.01
cutoff = 10

net = filter_lowpass(cutoff, dt)
net.add_input(0)
net.add_output(0,spiking=False)

model = net.compile(backend='numpy',dt=dt)

frequencies, output_peaks, phase_diff_deg = sample_frequency_response(model, low_hz=0.1, high_hz=1000, num_samples=10, plot=False, debug=True)

data = {'frequencies': frequencies, 'outputPeaks': output_peaks, 'phaseDiff': phase_diff_deg, 'dt': dt, 'cutoff': cutoff}

pickle.dump(data, open("data_lowpass.p", 'wb'))
