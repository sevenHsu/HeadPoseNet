import os
import cv2 as cv
import numpy as np
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


def show_loss_distribute(loss_dict, analysis_dir, snapshot_name):
    """

    :param loss_dict: {'angles':[[p,y,r],[],...],'degrees':[]}
    :param analysis_dir:directory for saving image
    :param snapshot_name:model snapshot name
    :return:
    """
    plt.switch_backend('agg')

    detail = snapshot_name

    angles = np.array(loss_dict['angles']) * 180 / np.pi
    degrees_error = np.array(loss_dict['degree_error'])

    plt.subplots(figsize=(30, 10))

    # figure pitch,yaw,roll
    for i, name in enumerate(['Pitch', 'Yaw', 'Roll']):
        plt.subplot(1, 3, i+1)
        plt.xlim(-100, 105)
        plt.xticks([j for j in range(-100, 105, 20)], [j for j in range(-100, 105, 20)])
        plt.ylim(-100, 105)
        plt.yticks([j for j in range(-100, 105, 10)], [j for j in range(-100, 105, 10)])
        plt.scatter(angles[:, i], degrees_error, linewidths=0.2)
        plt.title(name + ":Loss distribution(" + detail + ")")
        plt.xlabel(name + ":GT")
        plt.ylabel(name + ":Loss(degree-error)")
        plt.grid()

    plt.savefig(os.path.join(analysis_dir, detail + '.png'))
