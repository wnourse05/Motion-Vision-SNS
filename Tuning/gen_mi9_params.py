from utilities import save_data, cutoff_fastest, reversal_ex, activity_range

def tune_mi9(cutoff):
    type = 'lowpass'
    name = 'Retina'
    params = {'cutoff': cutoff,
              'invert': False,
              'initialValue': activity_range,
              'bias': 0.0}

    data = {'name': name,
            'type': type,
            'params': params}

    filename = '../params_node_mi9.p'
    save_data(data, filename)

    conn_params = {'source': 'L3',
                   'g': activity_range / (reversal_ex - activity_range),
                   'reversal': reversal_ex}
    conn_filename = '../params_conn_mi9.p'

    save_data(conn_params, conn_filename)
