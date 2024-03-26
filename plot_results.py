import pickle
import matplotlib.pyplot as plt
import torch

data_test = pickle.load(open('test_no_train_mean.p', 'rb'))
data_train = pickle.load(open('train_no_train_mean.p', 'rb'))

def plot_data(data, title):
    _, indices = torch.sort(data['targets'])
    targets_sorted = data['targets'][indices]
    ccw_sorted = data['ccw'][indices]
    cw_sorted = data['cw'][indices]
    start = 0
    i = 0
    current = 0.1
    targets = []
    ccw_means = []
    ccw_lows = []
    ccw_highs = []
    cw_means = []
    cw_lows = []
    cw_highs = []
    while i < len(targets_sorted):
        target = targets_sorted[i]
        if current<target:
            targets.append(current)
            current = target

            ccw_set = ccw_sorted[start:i-1]
            cw_set = cw_sorted[start:i-1]
            start = i

            ccw_means.append(torch.mean(ccw_set))
            cw_means.append(torch.mean(cw_set))

            ccw_lows.append(torch.quantile(ccw_set, 0.05))
            cw_lows.append(torch.quantile(cw_set, 0.05))

            ccw_highs.append(torch.quantile(ccw_set, 0.95))
            cw_highs.append(torch.quantile(cw_set, 0.95))

        i += 1
    ccw_set = ccw_sorted[start:i - 1]
    cw_set = cw_sorted[start:i - 1]

    ccw_means.append(torch.mean(ccw_set))
    cw_means.append(torch.mean(cw_set))

    ccw_lows.append(torch.quantile(ccw_set, 0.05))
    cw_lows.append(torch.quantile(cw_set, 0.05))

    ccw_highs.append(torch.quantile(ccw_set, 0.95))
    cw_highs.append(torch.quantile(cw_set, 0.95))
    targets.append(0.5)

    plt.figure()
    plt.subplot(2,1,1)
    plt.scatter(targets_sorted, ccw_sorted, label='CCW', alpha=0.5)
    plt.scatter(targets_sorted, cw_sorted, label='CW', alpha=0.5)
    plt.legend()
    plt.title(title)
    plt.ylabel('Mean Response')
    plt.subplot(2,1,2)
    plt.plot(targets, ccw_means, color='C0', label='CCW')
    plt.fill_between(targets, ccw_lows, ccw_highs, color='C0', alpha=0.1)
    plt.plot(targets, cw_means, color='C1', label='CW')
    plt.fill_between(targets, cw_lows, cw_highs, color='C1', alpha=0.1)
    plt.legend()
    plt.ylabel('Mean Response')
    plt.xlabel('Velocity (rad/s')

plot_data(data_test, 'Test Set')
plot_data(data_train, 'Training Set')
plt.show()