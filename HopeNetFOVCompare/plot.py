import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import json

x_range = np.arange(-90, 90, 15)


def get_y(angles, degrees, index):
    degree_group = [[] for i in range(-90, 90, 15)]
    angles = np.array(angles)
    yaw = angles[:, index]

    degrees = np.array(degrees)
    degrees = degrees[:, index]
    for i in range(len(yaw)):
        if -90 <= yaw[i] <= 90:
            degree_group[np.digitize(yaw[i], x_range) - 1].append(abs(degrees[i]))
    y = np.array([i for i in map(lambda item: sum(item) / len(item), degree_group)])
    return y


if __name__ == "__main__":
    with open("hopenet.json", 'r') as f:  # y p r
        hopenet = json.load(f)
    with open("fov.json", 'r') as f:  # p y r
        fov = json.load(f)
    hope_y = get_y(hopenet['angles'], hopenet['degrees'], 0)
    fov_y = get_y(fov['angles'], fov['degrees'], 1)
    # fov_y[3:] = fov_y[3:]-np.array([2.5,7,11])
    x = np.arange(-82.5, 90, 15)
    plt.plot(x, hope_y, color='red', label='Euler Angle')
    plt.plot(x, fov_y, color='green', label='FOV')
    plt.xticks([i for i in range(-90,91,15)])
    plt.xlabel('Yaw Angle')
    plt.ylabel('Mean Degree Error (abs)')
    plt.legend(loc='upper right')
    plt.savefig('group_error_yaw.pdf', dpi=2)
