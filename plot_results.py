import pickle
import matplotlib.pyplot as plt

data_test = pickle.load(open('test_no_train_mean.p', 'rb'))
data_train = pickle.load(open('train_no_train_mean.p', 'rb'))

print()