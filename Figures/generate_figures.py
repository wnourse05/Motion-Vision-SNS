import matplotlib.pyplot as plt
import pickle
import numpy as np

def desired_performance():
    data = pickle.load(open('desiredPerformance.p', 'rb'))

    plt.figure()
    plt.plot(data['samplePts'], data['magnitude'])
    plt.xlabel('Rotational Velocity (rad/s)')
    plt.ylabel('Neural Average Magnitude')
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

    plt.figure()
    plt.title('Image Scaling Performance')
    plt.plot(scale, latency_near, label='Nearest Neighbor')
    plt.plot(scale, latency_area, label='Area')
    plt.axvline(x=mine, color='black', linestyle='--', label='[24x64]')
    plt.xlabel('Reduction Factor')
    plt.ylabel('Image Processing Latency (ms)')
    plt.xscale('log')
    plt.legend()

def toolbox_vs_torch():
    toolbox_cpu = pickle.load(open('vary_size_snstoolbox_cpu.p', 'rb'))
    toolbox_gpu = pickle.load(open('vary_size_snstoolbox_cuda.p', 'rb'))
    torch_cpu = pickle.load(open('vary_size_snstorch_cpu.p', 'rb'))
    torch_gpu = pickle.load(open('vary_size_snstorch_cuda.p', 'rb'))
    xaxis = np.zeros(9)
    for i in range(9):
        xaxis[i] = toolbox_cpu['rows'][i]*toolbox_cpu['cols'][i]*20
    plt.figure()
    mine = 24*64*20
    plt.errorbar(xaxis[0], toolbox_cpu['avg'][0], yerr=[[toolbox_cpu['avg'][0]-toolbox_cpu['low'][0]],[toolbox_cpu['upper'][0]-toolbox_cpu['avg'][0]]], marker='s', label='SNS-Toolbox (CPU)')
    plt.errorbar(xaxis[0], toolbox_gpu['avg'][0], yerr=[[toolbox_gpu['avg'][0]-toolbox_gpu['low'][0]],[toolbox_gpu['upper'][0]-toolbox_gpu['avg'][0]]], marker='s', label='SNS-Toolbox (GPU)')
    plt.errorbar(xaxis, torch_cpu['avg'], yerr=[torch_cpu['avg'].numpy()-torch_cpu['low'].numpy(),torch_cpu['upper'].numpy()-torch_cpu['avg'].numpy()], marker='s', label='snsTorch (CPU)')
    plt.errorbar(xaxis[:-1], torch_gpu['avg'][:-1], yerr=[torch_gpu['avg'][:-1].numpy()-torch_gpu['low'][:-1].numpy(),torch_gpu['upper'][:-1].numpy()-torch_gpu['avg'][:-1].numpy()], marker='s', label='snsTorch (GPU)')
    plt.axvline(x=mine, color='black', linestyle='--', label='[24x64]')
    plt.xlabel('Number of Neurons')
    plt.ylabel('Time per Step (ms)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()

if __name__ == "__main__":
    # desired_performance()
    # camera_latency()
    toolbox_vs_torch()
    plt.show()