import pickle

type = 'lowpass'
name = 'Retina'
params = {'membraneCapacitance': 5.0}

data = {'name': name,
        'type': type,
        'params': params}

pickle.dump(data, open('params_neuron_retina.p', 'wb'))
