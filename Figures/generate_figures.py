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

if __name__ == "__main__":
    desired_performance()
    camera_latency()
    plt.show()