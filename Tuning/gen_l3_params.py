from utilities import save_data, cutoff_fastest, activity_range, synapse_target

def tune_l3(cutoff):
    type = 'lowpass'
    name = 'L3'
    params = {'cutoff': cutoff,
              'invert': True,
              'initialValue': activity_range,
              'bias': 0.0}

    data = {'name': name,
            'type': type,
            'params': params}

    filename = '../params_node_l3.p'
    save_data(data, filename)

    target_center = 0.0
    target_middle = 1.0 * activity_range
    target_outer = 17/16 * activity_range
    outer_conductance, outer_reversal = synapse_target(target_outer, activity_range)
    middle_conductance, middle_reversal = synapse_target(target_middle, activity_range)
    center_conductance, center_reversal = synapse_target(target_center, activity_range)
    g = {'outer': outer_conductance,
         'middle': middle_conductance,
         'center': center_conductance}
    reversal = {'outer': outer_reversal,
                'middle': middle_reversal,
                'center': center_reversal}
    conn_params = {'source': 'Retina',
                   'g': g,
                   'reversal': reversal}
    conn_filename = '../params_conn_l3.p'

    save_data(conn_params, conn_filename)
