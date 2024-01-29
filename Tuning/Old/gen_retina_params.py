from utilities import save_data

def tune_retina(cutoff, save=True):
    type = 'lowpass'
    name = 'Retina'
    params = {'cutoff': cutoff,
              'invert': False,
              'initialValue': 0.0,
              'bias': 0.0}

    data = {'name': name,
            'type': type,
            'params': params}

    filename = '../params_node_retina.p'
    if save:
        save_data(data, filename)

    return data