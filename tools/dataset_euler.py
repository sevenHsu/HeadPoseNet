import os
import torch
import numpy as np
from tools import utils
from PIL import Image, ImageFilter
from torch.utils.data.dataset import Dataset


def get_list_from_filenames(file_path):
    # input:    relative path to .txt file with file names
    # output:   list of relative path names
    with open(file_path) as f:
        lines = f.read().splitlines()
    return lines


class Pose300wlpEuler(Dataset):
    # Head pose from 300W-LP dataset
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        mat_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)
        x_min = min(pt2d[0, :])
        y_min = min(pt2d[1, :])
        x_max = max(pt2d[0, :])
        y_max = max(pt2d[1, :])

        # k = 0.2 to 0.40
        k = np.random.random_sample() * 0.2 + 0.2
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        img.show()
        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0] * 180 / np.pi
        yaw = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi

        # Flip?
        rnd = np.random.random_sample()
        if rnd < 0.5:
            yaw = -yaw
            roll = -roll
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Blur?
        rnd = np.random.random_sample()
        if rnd < 0.05:
            img = img.filter(ImageFilter.BLUR)

        # Bin values

        bins = np.array(range(-99, 102, 3))
        binned_pose = np.digitize([yaw, pitch, roll], bins) - 1

        # Get target tensors
        labels = binned_pose
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        # RGB2BGR
        img = img[np.array([2, 1, 0]), :, :]

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # 168,967
        return self.length


class Pose300wlpRdsEuler(Dataset):
    # 300W-LP dataset with random down sampling
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        mat_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)
        x_min = min(pt2d[0, :])
        y_min = min(pt2d[1, :])
        x_max = max(pt2d[0, :])
        y_max = max(pt2d[1, :])

        # k = 0.2 to 0.40
        k = np.random.random_sample() * 0.2 + 0.2
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)
        pitch = pose[0] * 180 / np.pi
        yaw = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi

        ds = 1 + np.random.randint(0, 4) * 5
        original_size = img.size
        img = img.resize((img.size[0] / ds, img.size[1] / ds), resample=Image.NEAREST)
        img = img.resize((original_size[0], original_size[1]), resample=Image.NEAREST)

        # Flip?
        rnd = np.random.random_sample()
        if rnd < 0.5:
            yaw = -yaw
            roll = -roll
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Blur?
        rnd = np.random.random_sample()
        if rnd < 0.05:
            img = img.filter(ImageFilter.BLUR)

        # Bin values
        bins = np.array(range(-99, 102, 3))
        binned_pose = np.digitize([yaw, pitch, roll], bins) - 1

        # Get target tensors
        labels = binned_pose
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        # RGB2BGR
        img = img[np.array([2, 1, 0]), :, :]

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # 168,967
        return self.length


class AFLW2000Euler(Dataset):
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        filename_list = get_list_from_filenames(filename_path)

        self.X_train = filename_list
        self.y_train = filename_list
        self.image_mode = image_mode
        self.length = len(filename_list)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index] + self.img_ext))
        img = img.convert(self.image_mode)
        mat_path = os.path.join(self.data_dir, self.y_train[index] + self.annot_ext)

        # Crop the face loosely
        pt2d = utils.get_pt2d_from_mat(mat_path)

        x_min = min(pt2d[0, :])
        y_min = min(pt2d[1, :])
        x_max = max(pt2d[0, :])
        y_max = max(pt2d[1, :])

        k = 0.20
        x_min -= 2 * k * abs(x_max - x_min)
        y_min -= 2 * k * abs(y_max - y_min)
        x_max += 2 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # We get the pose in radians
        pose = utils.get_ypr_from_mat(mat_path)
        # And convert to degrees.
        pitch = pose[0] * 180 / np.pi
        yaw = pose[1] * 180 / np.pi
        roll = pose[2] * 180 / np.pi
        # Bin values
        bins = np.array(range(-99, 102, 3))
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        if self.transform is not None:
            img = self.transform(img)

        # RGB2BGR
        img = img[np.array([2, 1, 0]), :, :]

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # 2,000
        return self.length


def get_list_from_dms(filename_path):
    image_name = []
    labels = []
    with open(filename_path, 'r') as fr:
        lines = fr.read().splitlines()
    for line in lines:
        line = line.split(' ')
        image_name.append(line[0])
        labels.append([float(j) for j in line[1:]])
    return image_name, labels


# Dataset for test dms
class DMS(Dataset):
    # Head pose from DMS-LP dataset
    def __init__(self, data_dir, filename_path, transform, img_ext='.jpg_face.jpg', annot_ext='.mat', image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.img_ext = img_ext
        self.annot_ext = annot_ext

        self.X_train, self.y_train = get_list_from_dms(filename_path)
        self.image_mode = image_mode
        self.length = len(self.X_train)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.X_train[index]))
        img = img.convert(self.image_mode)
        # pt2d = self.y_train[index]
        # x_min = pt2d[0]
        # y_min = pt2d[1]
        # x_max = pt2d[2]
        # y_max = pt2d[3]
        #
        # # k = 0.2 to 0.40
        # k = 0.2
        # x_min -= 0.6 * k * abs(x_max - x_min)
        # y_min -= 0.6 * k * abs(y_max - y_min)
        # x_max += 0.6 * k * abs(x_max - x_min)
        # y_max += 0.6 * k * abs(y_max - y_min)
        # img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        yaw = self.y_train[index][0]
        pitch = self.y_train[index][1]
        roll = self.y_train[index][2]

        bins = np.array(range(-99, 102, 3))
        labels = torch.LongTensor(np.digitize([yaw, pitch, roll], bins) - 1)
        cont_labels = torch.FloatTensor([yaw, pitch, roll])

        rnd = np.random.random_sample()
        if rnd < 0.05:
            img = img.filter(ImageFilter.BLUR)

        if self.transform is not None:
            img = self.transform(img)

        # RGB2BGR
        img = img[np.array([2, 1, 0]), :, :]

        return img, labels, cont_labels, self.X_train[index]

    def __len__(self):
        # 122,450
        return self.length


# predict dataset
class PredictData(Dataset):
    def __init__(self, data_dir, filename_list, transform, image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        if filename_list != '':
            self.images, self.bbox = get_list_from_dms(filename_list)
        else:
            self.images = os.listdir(self.data_dir)
            self.bbox = None
        self.length = len(self.images)
        self.image_mode = image_mode

    def __getitem__(self, index):
        image_path = os.path.join(self.data_dir, self.images[index])
        img = Image.open(image_path)
        img = img.convert(self.image_mode)

        # cropping images when self.bbox is not None
        if self.bbox is not None:
            if len(self.bbox[0]) == 4:
                x1 = self.bbox[index][0]
                y1 = self.bbox[index][1]
                x2 = self.bbox[index][2]
                y2 = self.bbox[index][3]
            else:
                x1 = self.bbox[index][1]
                y1 = self.bbox[index][2]
                x2 = self.bbox[index][3]
                y2 = self.bbox[index][4]

            w = x2 - x1
            h = y2 - y1

            k = 0.2
            x1 -= w * k * 2
            y1 -= h * k * 2
            x2 += w * k * 2
            y2 += h * k * 0.6

            img = img.crop((int(x1), int(y1), int(x2), int(y2)))

        # transform images
        if self.transform is not None:
            img = self.transform(img)

        return img, image_path

    def __len__(self):
        return self.length
