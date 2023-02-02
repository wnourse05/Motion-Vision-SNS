import pickle

c_fast = 1.0
activity_range = 1.0
reversal_ex = 5.0
reversal_in = -2.0
dt = 0.01
backend = 'numpy'

def save_data(data, filename):
    pickle.dump(data, open(filename, 'wb'))

def load_data(filename):
    data = pickle.load(open(filename, 'rb'))
    return data

def synapse_target(target, bias):
    if target > bias:
        reversal = reversal_ex
    else:
        reversal = reversal_in
    conductance = (bias - target)/(target - reversal)
    return conductance, reversal
