import pickle
import matplotlib.pyplot as plt
import numpy as np
from frequency_response import sample_frequency_response

from neurons_and_networks import filter_highpass
from sns_toolbox.renderer import render

dt = 0.01
cutoff = 100

net = filter_highpass(cutoff, dt)
net.add_input(1)
net.add_output(3,spiking=False)
render(net)
model = net.compile(backend='numpy',dt=dt)

frequencies, output_peaks, phase_diff_deg = sample_frequency_response(model, low_hz=1, high_hz=1000, num_samples=5, plot=False, debug=True)

data = {'frequencies': frequencies, 'outputPeaks': output_peaks, 'phaseDiff': phase_diff_deg, 'dt': dt, 'cutoff': cutoff}

pickle.dump(data, open("data_highpass.p", 'wb'))
