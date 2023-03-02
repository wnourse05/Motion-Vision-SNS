from utilities import save_data, reversal_ex, activity_range

def tune_tm9(cutoff):
    type = 'lowpass'
    name = 'Retina'
    params = {'cutoff': cutoff,
              'invert': False,
              'initialValue': activity_range,
              'bias': 0.0}

    data = {'name': name,
            'type': type,
            'params': params}

    filename = '../params_node_tm9.p'
    save_data(data, filename)

    conn_params = {'source': 'L3',
                   'g': activity_range / (reversal_ex - activity_range),
                   'reversal': reversal_ex}
    conn_filename = '../params_conn_tm9.p'

    save_data(conn_params, conn_filename)

    return data, conn_params
