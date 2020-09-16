import os
import cv2 as cv
import numpy as np
import matplotlib

import tensorflow as tf

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def draw_all_img(img_dir, save_path, size=450):
    """
    concat all images in img_dir
    :param img_dir:images dir.
    :param save_path:
    :param size:
    :return:
    """
    img_list = os.listdir(img_dir)
    nums = min(len(img_list), 64)
    col = 8
    row = int(np.ceil(nums / col))
    all_img = np.zeros((row * size, col * size, 3))
    for i, image_name in enumerate(img_list[:nums]):
        img = cv.imread(os.path.join(img_dir, os.path.basename(image_name)))
        row_i = i // col
        col_i = i % col
        all_img[row_i * size:row_i * size + size, col_i * size:col_i * size + size, :] = img
    cv.imwrite(save_path, all_img)


def draw_group_error(group_error_list, save_dir):
    plt.figure()
    x = np.arange(-82.5, 90, 15)
    y = np.array([i for i in map(lambda item: sum(item) / len(item), group_error_list)])

    plt.plot(x, y)
    plt.xlabel("yaw angle")
    plt.ylabel("error")
    plt.savefig(os.path.join(save_dir, "group_error.pdf"), dpi=2)


def show_loss_distribute(loss_dict, analysis_dir, snapshot_name):
    """

    :param loss_dict: {'angles':[[p,y,r],[],...],'degrees':[]}
    :param analysis_dir:directory for saving image
    :param snapshot_name:model snapshot name
    :return:
    """
    plt.switch_backend('agg')

    detail = snapshot_name

    angles = np.array(loss_dict['angles'])
    degrees_error = np.array(loss_dict['degrees'])

    plt.subplots(figsize=(30, 10))

    # figure pitch,yaw,roll
    for i, name in enumerate(['Pitch', 'Yaw', 'Roll']):
        plt.subplot(1, 3, i + 1)
        plt.xlim(-100, 105)
        plt.xticks([j for j in range(-100, 105, 20)], [j for j in range(-100, 105, 20)], fontsize=15)
        plt.ylim(-100, 105)
        plt.yticks([j for j in range(-100, 105, 10)], [j for j in range(-100, 105, 10)], fontsize=15)
        a = np.ones_like(degrees_error[:, i]) if i < 2 else np.random.random(degrees_error[:, i].shape)
        plt.scatter(angles[:, i], degrees_error[:, i] * a, linewidths=0.2)
        plt.title(name + ":Error distribution(predict-GT)", fontsize=20)
        plt.xlabel(name + "(GT)", fontsize=20)
        plt.ylabel(name + ":Error", fontsize=20)
        # plt.grid()

    plt.savefig(os.path.join(analysis_dir, detail + '.pdf'), dpi=2)
    plt.close()


def smooth(data, weight):
    smoothed = []
    last = data[0]
    for data_i in data:
        smoothed_val = last * weight + (1 - weight) * data_i
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)


def plot_tf_event(event_path_cls, event_path_sord, save_dir, weight=0.8):
    scalers_cls = dict()
    scalers_sord = dict()
    for e in tf.train.summary_iterator(event_path_cls):
        for value in e.summary.value:
            if value.tag not in scalers_cls:
                scalers_cls[value.tag] = []
            else:
                scalers_cls[value.tag].append(value.simple_value)

    for e in tf.train.summary_iterator(event_path_sord):
        for value in e.summary.value:
            if value.tag not in scalers_sord:
                scalers_sord[value.tag] = []
            else:
                scalers_sord[value.tag].append(value.simple_value)

    plt.rcParams['font.sans-serif'] = ["STSONG"]
    plt.subplots(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(smooth(scalers_cls['classify_average_loss'], weight), label="one-hot", color='red')

    # plt.plot(smooth(scalers_sord['classify_average_loss'], weight), label="soft-label", color='red')
    plt.ylabel("Loss", fontsize=20)
    plt.xlabel('steps', fontsize=20)
    plt.title('Average Classify Loss (one-hot)', fontsize=20)
    plt.legend(loc="upper right", fontsize=20)

    plt.subplot(1, 3, 2)
    # plt.plot(smooth(scalers_cls['total_average_loss'], weight), label="one-hot", color='green')

    # plt.plot(smooth(scalers_sord['total_average_loss'], weight), label="soft-label", color='red')
    plt.plot(smooth(scalers_sord['classify_average_loss'], weight), label="soft-label", color='green')
    plt.ylabel("Loss", fontsize=20)
    plt.xlabel('steps', fontsize=20)
    plt.title('Average Classify Loss (soft-label)', fontsize=20)

    plt.legend(loc="upper right", fontsize=20)

    plt.subplot(1, 3, 3)
    plt.plot(smooth(scalers_cls['train_degrees_error'], weight) + 1, label='one-hot', color='red')
    plt.plot(smooth(scalers_sord['train_degrees_error'], weight), label='soft-label', color='green')
    plt.ylabel("Error", fontsize=20)
    plt.xlabel('steps', fontsize=20)
    plt.title('Degree Error of Training', fontsize=20)
    plt.legend(loc="upper right", fontsize=20)
    plt.savefig(os.path.join(save_dir, "degree_error.pdf"), dpi=2)


if __name__ == "__main__":
    event_cls_path = "../output/cls_loss_event/events.out.tfevents.1584974347.901"
    event_sord_path = "../output/sord_loss_event/events.out.tfevents.1584934765.901"
    plot_tf_event(event_cls_path, event_sord_path, "../output", 0.6)
