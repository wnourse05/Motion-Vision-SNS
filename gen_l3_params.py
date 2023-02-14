from utilities import save_data, cutoff_fastest, activity_range

type = 'lowpass'
name = 'L3'
params = {'cutoff': cutoff_fastest,
          'invert': True,
          'initialValue': activity_range}

data = {'name': name,
        'type': type,
        'params': params}

filename = 'params_node_l3.p'

save_data(data, filename)
