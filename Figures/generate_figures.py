import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch
def state_to_vel(state):
    """
    Inverse of vel_to_state, convert a neural state to its corresponding velocity
    :param state: Neural state
    :return: Velocity
    """
    vel = ((1 - state) / (1 - 0.1)) * 0.4 + 0.1
    return vel

def desired_performance():
    data = pickle.load(open('desiredPerformance.p', 'rb'))
    print(data['magnitude'])
    print(state_to_vel(data['magnitude']))
    plt.figure()
    plt.plot(data['samplePts'], data['magnitude'], marker='s')
    plt.xlabel('Rotational Velocity (rad/s)')
    plt.ylabel('Neural Average Magnitude')
    plt.yscale('log')
    plt.title('Desired Neural Magnitude Across Rotational Velocity')

def camera_latency():
    full_row = 1232
    full_col = 3280
    rows = np.array([1232, 616, 308, 154, 77, 38, 24, 19])
    cols = np.array([3280, 1640, 820, 410, 205, 102, 64, 51])
    scale = full_row/rows
    mine = full_row/24
    latency_near = np.array([4, 4, 5, 5, 4, 4, 6, 4])
    latency_area = np.array([4, 5, 5, 7, 7, 22, 25, 26])

    # plt.figure()
    # plt.title('Image Scaling Performance')
    plt.plot(scale, latency_near, label='Nearest Neighbor')
    plt.plot(scale, latency_area, label='Area')
    plt.axvline(x=mine, color='black', linestyle='--', label='[24x64]')
    plt.xlabel('Reduction Factor')
    plt.ylabel('Image Processing Latency (ms)')
    plt.xscale('log')
    # plt.yscale('log')
    plt.legend()

def toolbox_vs_torch():
    toolbox_cpu = pickle.load(open('vary_size_snstoolbox_cpu.p', 'rb'))
    toolbox_gpu = pickle.load(open('vary_size_snstoolbox_cuda.p', 'rb'))
    torch_cpu = pickle.load(open('vary_size_snstorch_cpu.p', 'rb'))
    torch_gpu = pickle.load(open('vary_size_snstorch_cuda.p', 'rb'))
    xaxis = np.zeros(9)
    for i in range(9):
        xaxis[i] = toolbox_cpu['rows'][i]*toolbox_cpu['cols'][i]*20
    # plt.figure()
    mine = 24*64*20
    # plt.errorbar(xaxis[0], toolbox_cpu['avg'][0], yerr=[[toolbox_cpu['avg'][0]-toolbox_cpu['low'][0]],[toolbox_cpu['upper'][0]-toolbox_cpu['avg'][0]]], marker='s', label='SNS-Toolbox (CPU)')
    # plt.errorbar(xaxis[0], toolbox_gpu['avg'][0], yerr=[[toolbox_gpu['avg'][0]-toolbox_gpu['low'][0]],[toolbox_gpu['upper'][0]-toolbox_gpu['avg'][0]]], marker='s', label='SNS-Toolbox (GPU)')
    # plt.errorbar(xaxis, torch_cpu['avg'], yerr=[torch_cpu['avg'].numpy()-torch_cpu['low'].numpy(),torch_cpu['upper'].numpy()-torch_cpu['avg'].numpy()], marker='s', label='snsTorch (CPU)')
    # plt.errorbar(xaxis[:-1], torch_gpu['avg'][:-1], yerr=[torch_gpu['avg'][:-1].numpy()-torch_gpu['low'][:-1].numpy(),torch_gpu['upper'][:-1].numpy()-torch_gpu['avg'][:-1].numpy()], marker='s', label='snsTorch (GPU)')
    plt.plot(xaxis, torch_cpu['avg'], linestyle=':', color='C2', label='CPU')
    plt.fill_between(xaxis, torch_cpu['low'], torch_cpu['upper'], linestyle=':', color='C2', alpha=0.5)
    plt.plot(xaxis[:-1], torch_gpu['avg'][:-1], color='C3', label='GPU')
    plt.fill_between(xaxis[:-1], torch_gpu['low'][:-1], torch_gpu['upper'][:-1], color='C3', alpha=0.5)
    plt.axvline(x=mine, color='black', linestyle='--', label='[24x64]')
    plt.xlabel('Number of Neurons')
    plt.ylabel('Time per Step (s)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()

def headless_performance():
    data = pickle.load(open('headless_profile.p', 'rb'))
    times = data['rawTimes']
    median = torch.median(times)
    mean = torch.mean(times)
    (mode,_) = torch.mode(times)
    low = torch.quantile(times,0.05)
    high = torch.quantile(times, 0.95)
    print('95th: %.4f'%high)
    print('Median: %.4f'%median)
    print('Mean: %.4f'%mean)
    print('Mode: %.4f'%mode)
    x = np.arange(0,len(times))

    # plt.figure()
    # plt.scatter(x,times, label='Times')
    # plt.axhline(y=(1/30)/12*1000, color='red', label='12')
    # plt.axhline(y=(1/30)/13*1000, color='orange', label='13')
    # plt.axhline(y=(1/30)/14*1000, color='green', label='14')
    # plt.axhline(y=median, color='black', label='Median')
    # plt.axhline(y=mode, linestyle='-.', color='black', label='Mode')
    # plt.axhline(y=mean, linestyle=(5, (10, 3)), color='black', label='Mean')
    # plt.axhline(y=low, linestyle=':', color='black', label='5%')
    # plt.axhline(y=high, linestyle='--', color='black', label='95%')
    # plt.xlabel('Trial')
    # plt.ylabel('Step Time (ms)')
    # plt.yscale('log', base=10)
    # plt.legend()

    # plt.figure()
    # plt.scatter(x, times, label='Times')
    # plt.axhline(y=(1/30)/12*1000, color='red', label='12')
    # plt.axhline(y=(1/30)/13*1000, color='orange', label='13')
    # plt.axhline(y=(1/30)/14*1000, color='green', label='14')
    # plt.axhline(y=median, color='black', label='Median')
    # plt.axhline(y=mode, linestyle='-.', color='black', label='Mode')
    # plt.axhline(y=mean, linestyle=(5, (10, 3)), color='black', label='Mean')
    # plt.axhline(y=low, linestyle=':', color='black', label='5%')
    # plt.axhline(y=high, linestyle='--', color='black', label='95%')
    # plt.xlabel('Trial')
    # plt.ylabel('Step Time (ms)')
    # # plt.yscale('log', base=10)
    # plt.ylim([2.2,3])
    # plt.legend()

    # counts, bins = np.histogram(times, 10)
    # plt.figure()
    # plt.title('Jetson Orin Nano Step Time')
    # plt.hist(bins[:-1], bins, weights=counts)
    plt.hist(np.clip(times,0,3), bins=100, density=False, label="Times")
    plt.axvline(x=(1 / 30) / 12 * 1000, color='red', label='12 steps/frame')
    plt.axvline(x=(1/30)/13*1000, color='orange', label='13 steps/frame')
    plt.axvline(x=(1/30)/14*1000, color='green', linestyle=':', label='14 steps/frame')
    plt.axvline(x=high, linestyle='--', color='black', label='95%')
    plt.xlabel('Step Time (ms)')
    plt.ylabel('Number of Trials')
    plt.legend()


if __name__ == "__main__":
    plt.figure()
    # desired_performance()
    plt.subplot(1,3,1)
    camera_latency()
    plt.subplot(1,3,2)
    toolbox_vs_torch()
    plt.subplot(1,3,3)
    headless_performance()
    plt.show()