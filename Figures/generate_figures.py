import matplotlib.pyplot as plt
import pickle

def desired_performance():
    data = pickle.load(open('../Data/desiredPerformance.p', 'rb'))

    plt.figure()
    plt.plot(data['samplePts'], data['magnitude'])
    plt.xlabel('Rotational Velocity (rad/s)')
    plt.ylabel('Neural Average Magnitude')
    plt.title('Desired Neural Magnitude Across Rotational Velocity')

if __name__ == "__main__":
    desired_performance()
    plt.show()