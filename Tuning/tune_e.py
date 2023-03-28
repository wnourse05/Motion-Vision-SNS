from utilities import save_data, reversal_ex, activity_range

def tune_e(cutoff):
    params = {'cutoff': cutoff,
              'invert': False,
              'initialValue': activity_range,
              'bias': 0.0,
              'g': activity_range / (reversal_ex - activity_range),
              'reversal': reversal_ex}

    return params
