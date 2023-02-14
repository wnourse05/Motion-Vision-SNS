from utilities import save_data, cutoff_fastest

type = 'lowpass'
name = 'Retina'
params = {'cutoff': cutoff_fastest,
          'invert': False,
          'initialValue': 0.0,
          'bias': 0.0}

data = {'name': name,
        'type': type,
        'params': params}

filename = 'params_node_retina.p'

save_data(data, filename)
