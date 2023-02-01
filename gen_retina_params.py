from utilities import save_data

type = 'lowpass'
name = 'Retina'
params = {'membraneCapacitance': 5.0}

data = {'name': name,
        'type': type,
        'params': params}

filename = 'params_neuron_retina.p'

save_data(data, filename)
