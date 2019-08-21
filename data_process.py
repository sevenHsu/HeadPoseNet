# -*- coding:utf-8 -*-
import os
import tqdm
import json
import argparse
import cv2 as cv
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import trange
from tools.utils import mkdir
from shutil import copyfile, move
from tools.rotation import Rotation as R


def args_parse():
    parser = argparse.ArgumentParser(description='draw face bbox on origin images and save or crop face and save')
    parser.add_argument('--data_dir', dest='data_dir', help='origin images directory path',
                        default='', type=str)
    parser.add_argument('--filename_list', dest='filename_list', help='image name and face confident score and bbox',
                        default='', type=str)
    parser.add_argument('--mode', dest='mode', help='draw face bbox or crop',
                        default='draw', type=str)
    parser.add_argument('--save_dir', dest='save_dir', help='directory path for save drawn images or cropped images',
                        default='', type=str)
    parser.add_argument('--loosely', dest='loosely', help='loosely for crop images',
                        default=None, type=float)
    parser.add_argument('--ratio', dest='ratio', help='ratio for resize images',
                        default=0.5625, type=float)
    parser.add_argument('--save_list', dest='save_list', help='file for saving cropped bbox',
                        default='', type=str)
    parser.add_argument('--score_threshold', dest='score_threshold', help='threshold for filter faces',
                        default=0.0, type=float)
    parser.add_argument('--groups', dest='groups', help='groups for split dms data',
                        default=0, type=int)
    parser.add_argument('--pose_list', dest='pose_list', help='image name and pose pitch,yaw,roll',
                        default='', type=str)
    parser.add_argument('--limit', dest='limit', default=20, type=int)
    args = parser.parse_args()

    return args


def draw_face(data_dir, filename_list, save_dir):
    """
    draw face box on images
    :param data_dir:
    :param filename_list:
    :param save_dir:
    :return:
    """
    mkdir(save_dir)
    if filename_list != '':
        with open(filename_list, 'r') as fr:
            lines = fr.read().splitlines()
        for line in tqdm.tqdm(lines):
            line = line.split(' ')

            image_name = line[0]
            if len(line) == 5:
                x1 = float(line[1])
                y1 = float(line[2])
                x2 = float(line[3])
                y2 = float(line[4])
            else:
                x1 = float(line[2])
                y1 = float(line[3])
                x2 = float(line[4])
                y2 = float(line[5])

            image_path = os.path.join(data_dir, image_name)
            img = cv.imread(image_path)
            img = cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255))
            cv.imwrite(os.path.join(save_dir, os.path.basename(image_name)), img)
    else:
        txt_list = os.listdir(os.path.join(data_dir, 'bbox'))
        for txt in tqdm.tqdm(txt_list):
            image_name = txt[:-3] + 'jpg'
            with open(os.path.join(data_dir, 'bbox/' + txt), 'r') as fw:
                bbox = fw.read().splitlines()
            line = bbox[0].split(' ')

            if len(line) == 5:
                x1 = float(line[1])
                y1 = float(line[2])
                x2 = float(line[3])
                y2 = float(line[4])
            else:
                x1 = float(line[0])
                y1 = float(line[1])
                x2 = float(line[2])
                y2 = float(line[3])

            image_path = os.path.join(data_dir, 'bg_imgs/' + image_name)
            img = cv.imread(image_path)
            img = cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255))
            cv.imwrite(os.path.join(save_dir, os.path.basename(image_name)), img)


def get_croped_bbox(bbox_i, loosely):
    x1 = bbox_i[0]
    y1 = bbox_i[1]
    x2 = bbox_i[2]
    y2 = bbox_i[3]

    w = x2 - x1
    h = y2 - y1

    x1 -= loosely * w
    y1 -= loosely * h
    x2 += loosely * w
    y2 += loosely * h
    margin = np.abs(x2 - x1 - y2 + y1) / 2
    if (x2 - x1) > (y2 - y1):
        y1 -= margin
        y2 += margin
    else:
        x1 -= margin
        x2 += margin
    return x1, y1, x2, y2


def get_croped_quat(pose_i):
    pose = [float(i) for i in pose_i]
    right_dcm = R.euler2dcm(pose[0], pose[1], pose[2])
    left_dcm = R.transform_dcm(right_dcm)
    quat = R.dcm2quat(left_dcm)

    return quat


def crop_face(data_dir, bbox_file, pose_file, loosely, save_dir, score_threshold=0.7):
    """

    :param data_dir:
    :param bbox_file:
    :param pose_file:
    :param loosely:
    :param save_dir:
    :param score_threshold:
    :return:
    """
    mkdir(os.path.join(save_dir, 'bg_imgs'))
    mkdir(os.path.join(save_dir, 'bbox'))
    mkdir(os.path.join(save_dir, 'info'))

    bbox_dict = dict()

    with open(bbox_file, 'r') as fr:
        bbox = fr.read().splitlines()
    with open(pose_file, 'r') as fr1:
        pose = fr1.read().splitlines()
    for bbox_i in bbox:
        bbox_i = bbox_i.split(' ')
        bbox_dict[bbox_i[0]] = [float(i) for i in bbox_i[1:]]

    str_len = len(str(len(pose)))

    for i, pose_i in enumerate(tqdm.tqdm(pose)):
        pose_i = pose_i.split(' ')

        image_name = pose_i[0]

        bbox_i = bbox_dict[image_name]

        score = float(bbox_i[0])
        if score < score_threshold:
            continue
        x1, y1, x2, y2 = get_croped_bbox(bbox_i[1:], loosely)

        x, y, z, w = get_croped_quat(pose_i[1:])

        image_path = os.path.join(data_dir, image_name)
        img = Image.open(image_path)
        img.convert('RGB')
        img = img.crop((int(x1), int(y1), int(x2), int(y2)))
        img.save(os.path.join(save_dir, 'bg_imgs/' + os.path.basename(image_name)[:-4] + '_' + str(i).zfill(str_len) + '.jpg'))

        new_x1 = float(bbox_i[1]) - x1
        new_y1 = float(bbox_i[2]) - y1
        new_x2 = float(bbox_i[3]) - x1
        new_y2 = float(bbox_i[4]) - y1
        with open(os.path.join(save_dir, 'bbox/' + os.path.basename(image_name)[:-4] + '_' + str(i).zfill(str_len) + '.txt'), 'w') as fw:
            fw.write("%.3f %.3f %.3f %.3f\n" % (new_x1, new_y1, new_x2, new_y2))

        with open(os.path.join(save_dir, 'info/' + os.path.basename(image_name)[:-4] + '_' + str(i).zfill(str_len) + '.txt'), 'w') as fw1:
            fw1.write("%.5f %.5f %.5f %.5f\n" % (x, y, z, w))

        with open(os.path.join(save_dir, 'angles/' + os.path.basename(image_name)[:-4] + '_' + str(i).zfill(str_len) + '.txt'), 'w') as fw2:
            fw2.write("%s %s %s\n" % (pose_i[1], pose_i[2], pose_i[3]))


def resize_image(data_dir, filename_list, ratio, save_dir):
    """
    resize dms data to 16:9
    :param data_dir:
    :param filename_list:
    :param ratio:
    :param save_dir:
    :return:
    """
    with open(filename_list, 'r') as fr:
        lines = fr.read().splitlines()
    for i, img_name in enumerate(tqdm.tqdm(lines)):
        img = cv.imread(os.path.join(data_dir, img_name))
        h, w = img.shape[:2]
        img = cv.resize(img, (w, int(w * ratio)))
        cv.imwrite(os.path.join(save_dir, str(i).zfill(6) + '.jpg'), img)


def dms_euler2quat(data_dir, pose_list, save_dir):
    """
    transform dms dataset to quaternions from euler
    :param data_dir:
    :param pose_list:
    :param save_dir:
    :return:
    """
    # 将dms数据欧拉角转四元组，作为人脸3D模型的初始pose
    with open(pose_list, 'r') as fr:
        lines = fr.read().splitlines()
    for line in tqdm.tqdm(lines):
        line = line.split(' ')
        image_name = line[0]
        pitch = float(line[1])
        yaw = float(line[2])
        roll = float(line[3])
        right_dcm = R.euler2dcm(pitch, yaw, roll)
        left_dcm = R.transform_dcm(right_dcm)
        quat = R.dcm2quat(left_dcm)
        # img = cv.imread(os.path.join(data_dir, image_name))
        # img = padding_image(img)
        # cv.imwrite(os.path.join(save_dir + '/bg_imgs', image_name), img)
        with open(os.path.join(save_dir + '/info', image_name[:-3] + 'txt'), 'w') as fw:
            fw.write("%.5f %.5f %.5f %.5f" % (quat[0], quat[1], quat[2], quat[3]))


def pub_dataset2quat(data_dir, filename_list, save_dir):
    """
    transform public dataset to quaternions from euler
    :param data_dir:
    :param filename_list:
    :param save_dir:
    :return:
    """
    with open(filename_list, 'r') as fr:
        lines = fr.read().splitlines()
    lines_len = len(lines)
    fill_len = len(str(lines_len))
    for i in trange(lines_len):
        file_name = lines[i]
        image_path = os.path.join(data_dir, file_name + '.jpg')
        mat_path = os.path.join(data_dir, file_name + '.mat')
        mat = sio.loadmat(mat_path)

        pt2d = mat['pt2d']
        x_min = min(pt2d[0, :])
        y_min = min(pt2d[1, :])
        x_max = max(pt2d[0, :])
        y_max = max(pt2d[1, :])

        pose = mat['Pose_Para'][0][:3]
        pitch = pose[0]
        yaw = pose[1]
        roll = pose[2]

        new_file_name = str(i).zfill(fill_len)
        copyfile(image_path, os.path.join(save_dir + '/bg_imgs', new_file_name + '.jpg'))

        right_dcm = R.euler2dcm(pitch, yaw, roll)
        left_dcm = R.transform_dcm(right_dcm)
        quat = R.dcm2quat(left_dcm)
        with open(os.path.join(save_dir + '/info', new_file_name + '.txt'), 'w') as fw_quat:
            fw_quat.write("%.5f %.5f %.5f %.5f" % (quat[0], quat[1], quat[2], quat[3]))

        with open(os.path.join(save_dir + '/bbox', new_file_name + '.txt'), 'w') as fw_bbox:
            fw_bbox.write("%.2f %.3f %.3f %.3f" % (x_min, y_min, x_max, y_max))
        with open(os.path.join(save_dir + '/angles', new_file_name + '.txt'), 'w') as fw_angle:
            fw_angle.write('%.5f %.5f %.5f\n' % (pitch, yaw, roll))


def split_dataset(data_dir, save_dir, groups):
    """
    split dms dataset to several groups
    :param data_dir:
    :param save_dir:
    :param groups:
    :return:
    """
    mkdir(save_dir)
    filelist = os.listdir(os.path.join(data_dir, 'bg_imgs'))
    mount = len(filelist)
    group_mount = int(mount // groups)
    for i in trange(0, groups):
        main_dir = os.path.join(save_dir, data_dir + '_' + str(i))
        mkdir(main_dir)
        mkdir(os.path.join(main_dir, 'bg_imgs'))
        mkdir(os.path.join(main_dir, 'info'))
        mkdir(os.path.join(main_dir, 'bbox'))
        mkdir(os.path.join(main_dir, 'angles'))
        for file in tqdm.tqdm(filelist[i * group_mount:min((i + 1) * group_mount, mount)]):
            id = file[:-4]
            copyfile(os.path.join(data_dir, 'bg_imgs/' + id + '.jpg'), os.path.join(main_dir, 'bg_imgs/' + id + '.jpg'))
            copyfile(os.path.join(data_dir, 'info/' + id + '.txt'), os.path.join(main_dir, 'info/' + id + '.txt'))
            copyfile(os.path.join(data_dir, 'bbox/' + id + '.txt'), os.path.join(main_dir, 'bbox/' + id + '.txt'))
            copyfile(os.path.join(data_dir, 'angles/' + id + '.txt'), os.path.join(main_dir, 'angles/' + id + '.txt'))


def padding_image(img):
    h, w = img.shape[:2]
    size = max(h, w)
    y, x = int((size - h) // 2), int((size - w) // 2)
    padded_img = np.zeros((size, size, 3))
    padded_img[y:y + h, x:x + w, :] = img
    return padded_img


def dms_euler_analysis(filename_list):
    """
    analysis dms dataset euler angle distribute
    :param filename_list:
    :return:
    """
    angle_list = np.array(range(-99, 102, 33))
    count = np.zeros(6)
    with open(filename_list, 'r') as fr:
        lines = fr.read().splitlines()
    for line in tqdm.tqdm(lines):
        line = line.split(' ')
        yaw = float(line[2]) * 180 / np.pi
        count[np.digitize(yaw, angle_list) - 1] += 1
    print(count)


def choice_faces(filename_list, save_list):
    """
    filter images with euler angel on support distribution
    :param filename_list:
    :param save_list:
    :return:
    """
    # mount = 100000
    angle_list = np.array(range(-99, 102, 33))
    # mount_ratio = np.array([0.25, 0.2, 0.05, 0.05, 0.2, 0.25])
    mount_list = [50, 4000, 2000, 2000, 3500, 20]

    with open(filename_list, 'r') as fr:
        lines = fr.read().splitlines()
    fw = open(save_list, 'w')
    for line in tqdm.tqdm(lines):
        line = line.split(' ')
        image_name = line[0]
        pitch = float(line[1])
        yaw = float(line[2])
        roll = float(line[3])

        yaw_degrees = yaw * 180 / np.pi

        if np.digitize(yaw_degrees, angle_list) == 1 and mount_list[0] > 0:
            mount_list[0] -= 1
        elif np.digitize(yaw_degrees, angle_list) == 2 and mount_list[1] > 0:
            mount_list[1] -= 1
        elif np.digitize(yaw_degrees, angle_list) == 3 and mount_list[2] > 0:
            mount_list[2] -= 1
        elif np.digitize(yaw_degrees, angle_list) == 4 and mount_list[3] > 0:
            mount_list[3] -= 1
        elif np.digitize(yaw_degrees, angle_list) == 5 and mount_list[4] > 0:
            mount_list[4] -= 1
        elif np.digitize(yaw_degrees, angle_list) == 6 and mount_list[5] > 0:
            mount_list[5] -= 1
        else:
            continue
        fw.write("%s %.3f %.3f %.3f\n" % (image_name, pitch, yaw, roll))


def collect_score(data_dir):
    """

    :param data_dir:
    :return:
    """
    plt.switch_backend('agg')
    dir_list = os.listdir(data_dir)
    dict_path_list = list()
    for dir in dir_list:
        collect_dir = os.path.join(data_dir, dir + '/collect_score')
        if not os.path.exists(collect_dir):
            continue
        json_names = os.listdir(collect_dir)
        for json_name in json_names:
            json_path = os.path.join(collect_dir, json_name)
            if os.path.exists(json_path):
                dict_path_list.append(json_path)
    color = ['red', 'black', 'green', 'blue', 'yellow', 'brown', 'gray']
    x = np.array(range(5, 181, 5))
    for i, path in enumerate(dict_path_list):
        with open(path, 'r') as fr:
            degree_error = np.array(json.load(fr)['degree_error'])
        mount = np.zeros(len(x))
        for j in range(len(x)):
            mount[j] = sum(degree_error < x[j])
        y = mount / len(degree_error)
        plt.plot(x, y, c=color[i], label=os.path.basename(path)[:-5])
    plt.legend(loc='lower right', fontsize='x-small')
    plt.xlabel('degrees upper limit')
    plt.ylabel('accuracy')
    plt.xlim(40, 105)
    plt.ylim(0.4, 1.05)
    plt.xticks([j for j in range(5, 105, 5)], [j for j in range(5, 105, 5)])
    plt.yticks([j / 100 for j in range(40, 105, 5)], [j / 100 for j in range(40, 105, 5)])
    plt.title("accuracy under degree upper limit")
    plt.grid()
    plt.savefig(data_dir + '/collect_score.png')


def choice_dms_face(bbox_file, pose_file, save_list, limit):
    """

    :param bbox_file:
    :param pose_file:
    :param save_list:
    :param limit:
    :return:
    """
    id_dict = {}
    angles = np.array([0, 33, 66, 99])
    with open(bbox_file, 'r') as bbox_fr:
        bbox_list = bbox_fr.read().splitlines()

    bbox_dict = dict()
    for bbox_i in bbox_list:
        bbox_i = bbox_i.split(' ')
        bbox_dict[bbox_i[0]] = [float(i) for i in bbox_i[1:]]

    with open(pose_file, 'r') as pose_fr:
        pose_list = pose_fr.read().splitlines()

    with open(save_list, 'w') as fw:
        for pose_ii in tqdm.tqdm(pose_list):
            pose_i = pose_ii.split(' ')

            image_name = pose_i[0]

            id = image_name.split('/')[1]

            if id not in id_dict:
                id_dict[id] = [0, 0, 0, 0]

            score, x1, y1, x2, y2 = bbox_dict[image_name]
            if min((x2 - x1), (y2 - y1)) < 100:
                continue
            image_name_keywords = image_name.split('_')
            if 'glasses' not in image_name_keywords and 'nod' not in image_name_keywords and 'Visual-movement' not in image_name_keywords and np.random.randn() < 0.80:
                continue
            if id_dict[id][0] > limit:
                continue
            yaw = np.abs(float(pose_i[2]) * 180 / np.pi)
            pos = np.digitize(yaw, angles)
            if pos > 3:
                continue
            if id_dict[id][pos] > limit // 3:
                continue
            id_dict[id][0] += 1
            id_dict[id][pos] += 1
            fw.write(pose_ii + '\n')


def all_img(data_dir):
    dir_list = os.listdir(data_dir)
    for dir in dir_list:
        huge_eror_path = os.path.join(data_dir, dir + '/huge_error')
        if not os.path.exists(huge_eror_path):
            continue
        else:
            huge_eror_img_list = os.listdir(huge_eror_path)
            nums = min(100, len(huge_eror_img_list))
            rows = int(np.sqrt(nums))
            cols = int(np.ceil(nums / rows))
            all_img = np.zeros((rows * 300, cols * 300, 3))

            for k, img in enumerate(huge_eror_img_list[:nums]):
                i = k // rows
                j = k - (i * rows)
                img_path = os.path.join(huge_eror_path, img)
                img = cv.imread(img_path)
                img = cv.resize(img, (300, 300))
                all_img[j * 300:(j + 1) * 300, i * 300:(i + 1) * 300, :] = img
            cv.imwrite(data_dir + "/" + dir + '/huge_error.jpg', all_img)


if __name__ == '__main__':
    args = args_parse()
    if args.mode == 'draw':
        draw_face(args.data_dir, args.filename_list, args.save_dir)
    if args.mode == 'crop':
        crop_face(args.data_dir, args.filename_list, args.pose_list, args.loosely, args.save_dir, args.score_threshold)
    if args.mode == 'resize':
        resize_image(args.data_dir, args.filename_list, args.ratio, args.save_dir)
    if args.mode == 'analysis':
        dms_euler_analysis(args.filename_list)
    if args.mode == 'choice':
        choice_faces(args.filename_list, args.save_list)
    if args.mode == 'dms_quat':
        dms_euler2quat(args.data_dir, args.filename_list, args.save_dir)
    if args.mode == 'collect':
        collect_score(args.data_dir)
    if args.mode == 'split_data':
        split_dataset(args.data_dir, args.save_dir, args.groups)
    if args.mode == 'choice_dms':
        choice_dms_face(args.filename_list, args.pose_list, args.save_list, args.limit)
    if args.mode == 'all_img':
        all_img(args.data_dir)
