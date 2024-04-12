import pickle
import matplotlib.pyplot as plt
import torch

data_test = pickle.load(open('test_no_train_mean.p', 'rb'))
data_train = pickle.load(open('train_no_train_mean.p', 'rb'))
field_test = pickle.load(open('field_test_no_train_mean.p', 'rb'))
field_train = pickle.load(open('field_train_no_train_mean.p', 'rb'))
seed_1 = pickle.load(open('2024-03-29-1711743092.2438712_best.p', 'rb'))
seed_10 = pickle.load(open('2024-03-31-1711915538.720788_best.p', 'rb'))

def accuracy(data):
    ccw = data['ccw']
    cw = data['cw']
    num_trials = len(cw)
    num_correct = 0
    for i in range(num_trials):
        if ccw[i] > cw[i]:
            num_correct += 1
    print('%i Trials, %i Correct, %.4f Accuracy Ratio'%(num_trials, num_correct, num_correct/num_trials))

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
    pos_max = max(max([ccw_highs, cw_highs]))
    neg_max = abs(min(min([ccw_lows, cw_lows])))
    peak_max = max([pos_max,neg_max])
    peak_max = peak_max.item()

    # for i in range(len(ccw_highs)):
    #     ccw_means[i] /= peak_max
    #     ccw_lows[i] /= peak_max
    #     ccw_highs[i] /= peak_max
    #     cw_means[i] /= peak_max
    #     cw_lows[i] /= peak_max
    #     cw_highs[i] /= peak_max

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
    plt.ylabel('Mean Normalized Response')
    plt.xlabel('Velocity (rad/s')

# plot_data(data_test, 'No Field')
# plot_data(data_train, 'Training Set')
# plot_data(field_test, 'Field')
# plot_data(field_train, 'Training Set')
# accuracy(data_test)
# accuracy(data_train)
# accuracy(field_test)
# accuracy(field_train)
plot_data(seed_1, 'Seed: 1')
plot_data(seed_10, 'Seed: 10')
accuracy(seed_1)
accuracy(seed_10)

plt.show()