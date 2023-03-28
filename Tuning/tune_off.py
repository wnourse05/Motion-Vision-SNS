from utilities import synapse_target, activity_range

def tune_off(cutoff, div, ratio_enhance_off):
    g_transmit, rev_transmit = synapse_target(activity_range, 0.0)
    g = {'direct': (1 - ratio_enhance_off) * g_transmit,
        'enhance': ratio_enhance_off * g_transmit,
        'suppress': div - 1}
    reversal = {'direct': rev_transmit,
                'enhance': rev_transmit,
                'suppress': 0.0}

    params = {'cutoff': cutoff,
              'invert': False,
              'initialValue': 0.0,
              'bias': 0.0,
              'g': g,
              'reversal': reversal}

    return params