import pickle

def save_data(data, filename):
    pickle.dump(data, open(filename, 'wb'))

def load_data(filename):
    data = pickle.load(open(filename, 'rb'))
    return data