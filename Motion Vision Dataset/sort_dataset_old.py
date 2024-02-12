import pickle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

rng = np.random.default_rng(seed=0) # Random number generator

data = pickle.load(open('dataset.p', 'rb')) # Load data

images = np.moveaxis(data['data'],0,2)  # Pull image data, and reorder axes

labels = np.repeat(data['labels'], 2)    # Double the labels, because the dataset has left and right split
# i_sorted = np.argsort(labels)
# labels_sorted = labels[i_sorted]


num_samples = len(labels)    # Number of samples
print('%i Samples'%num_samples)
counts, _ = np.histogram(labels) # Number of each velocity available
bins = np.unique(labels) # velocity categories
num_categories = len(bins)
print('%i Categories'%num_categories)
ratio_test = 0.2

i_random = rng.choice(num_samples, num_samples) # randomized, non-repeating indices

# Test/Train breakdown
num_test = num_categories*round(ratio_test*num_samples/num_categories)
num_per_category = num_test/num_categories
num_train = num_samples - num_test
print('Test Size:  %i'%num_test)
print('Test Number per Category: %i'%num_per_category)
print('Train Size: %i'%num_train)

set_test = None
set_train = None
labels_test = None
labels_train = None

for i in range(num_samples):
    print('%i/%i'%(i+1,num_samples))
    vel = labels[i_random[i]]
    image = images[:,:,:,i_random[i]]
    name = 'gifs/%.2f_%i.gif'%(vel,i)
    frames = []
    for j in range(90):
        # print('j = %i'%j)
        if j == 0:
            frames = [Image.fromarray(image[:,:,j])]
        else:
            frames.append(Image.fromarray(image[:,:,j]))
    frames[0].save(name, save_all=True, append_images=frames[1:], duration=50, loop=0)
    if labels_test is None:
        print('Creating Test Set')
        labels_test = np.array([vel])
        set_test = image
    elif np.count_nonzero(labels_test == vel) < num_per_category:
        print('Adding to Test Set')
        labels_test = np.hstack([labels_test,vel])
        if len(set_test.shape) > 3:
            set_test = np.concatenate([set_test,image[...,None]],axis=3)
        else:
            set_test = np.stack([set_test,image], axis=3)
    else:
        if labels_train is None:
            print('Creating Training Set')
            labels_train = np.array([vel])
            set_train = image
        else:
            print('Adding to Training Set')
            labels_train = np.hstack([labels_train, vel])
            if len(set_train.shape) > 3:
                set_train = np.concatenate([set_train, image[..., None]], axis=3)
            else:
                set_train = np.stack([set_train, image], axis=3)

print(len(labels_test))
print(len(labels_train))

data_train = {'data':set_train, 'labels': labels_train}
data_test = {'data':set_test, 'labels': labels_test}

pickle.dump(data_test, open('set_test.p', 'wb'))
pickle.dump(data_train, open('set_train.p', 'wb'))

# plt.figure()
# plt.bar(bins, counts)
# print('Data Loaded')
# plt.show()