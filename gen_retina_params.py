from utilities import save_data, c_fast

type = 'lowpass'
name = 'Retina'
params = {'membraneCapacitance': c_fast}

data = {'name': name,
        'type': type,
        'params': params}

filename = 'params_neuron_retina.p'

save_data(data, filename)
