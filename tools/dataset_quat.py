import os
import torch
import numpy as np
from PIL import Image
from tools.utils import add_mask
from PIL import ImageFilter
from torch.utils.data.dataset import Dataset
from tools.utils import get_label_from_txt, get_attention_vector


def get_soft_label(cls_label, num_classes):
    """
    compute soft label replace one-hot label
    :param cls_label:ground truth class label
    :param num_classes:mount of classes
    :return:
    """

    # def metrix_fun(a, b):
    #     torch.IntTensor(a)
    #     torch.IntTensor(b)
    #     metrix_dis = (a - b) ** 2
    #     return metrix_dis
    def metrix_fun(a, b):
        a = a.type_as(torch.FloatTensor())
        b = b.type_as(torch.FloatTensor())
        metrix_dis = (torch.log(a) - torch.log(b)) ** 2
        return metrix_dis

    def exp(x):
        x = x.type_as(torch.FloatTensor())
        return torch.exp(x)

    rt = torch.IntTensor([cls_label])  # must be torch.IntTensor or torch.LongTensor
    rk = torch.IntTensor([idx for idx in range(1, num_classes + 1, 1)])
    metrix_vector = exp(-metrix_fun(rt, rk))
    return metrix_vector / torch.sum(metrix_vector)


# direct regression training dataset
class TrainDataSetReg(Dataset):
    # Head pose from 300W-LP dataset while using directly regression
    def __init__(self, data_dir, transform, image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform

        self.data_list = os.listdir(os.path.join(self.data_dir, 'bg_imgs'))

        self.image_mode = image_mode
        self.length = len(self.data_list)

    def __getitem__(self, index):
        base_name = self.data_list[index][:-4]
        img = Image.open(os.path.join(self.data_dir, 'bg_imgs/' + base_name + '.jpg'))
        img = img.convert(self.image_mode)

        # get face bbox
        bbox_path = os.path.join(self.data_dir, 'bbox/' + base_name + '.txt')

        pt2d = get_label_from_txt(bbox_path)
        x_min = pt2d[0]
        y_min = pt2d[1]
        x_max = pt2d[2]
        y_max = pt2d[3]

        # Crop the face loosely
        # k = 0 to 0.2
        k = np.random.random_sample() * 0.1
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # get pose quat
        quat_path = os.path.join(self.data_dir, 'info/' + base_name + '.txt')

        quat = get_label_from_txt(quat_path)

        # Blur?
        rnd = np.random.random_sample()
        if rnd < 0.05:
            img = img.filter(ImageFilter.BLUR)

        # Gray?
        rnd = np.random.random_sample()
        if rnd < 0.85 and base_name.find('ID') < 0:
            img = img.convert('L').convert('RGB')

        # Mask?
        rnd = np.random.random_sample()
        if rnd < 0.:
            img = add_mask(img)

        # Attention vector
        attention_vector = get_attention_vector(quat)
        vector_label = torch.FloatTensor(attention_vector)

        if self.transform is not None:
            img = self.transform(img)

        # RGB2BGR
        img = img[np.array([2, 1, 0]), :, :]

        return img, vector_label, os.path.join(self.data_dir, 'bg_imgs/' + base_name + '.jpg')

    def __len__(self):
        # 168,967
        return self.length


# classify and regression training dataset
class TrainDataSetCls(Dataset):
    # Head pose from 300W-LP dataset while using expectation of classify softmax results to regress
    def __init__(self, data_dir, transform, num_classes, image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.num_classes = num_classes
        self.bin_size = 198 // self.num_classes

        self.data_list = os.listdir(os.path.join(self.data_dir, 'bg_imgs'))

        self.image_mode = image_mode
        self.length = len(self.data_list)

    def __getitem__(self, index):
        base_name = self.data_list[index][:-4]
        img = Image.open(os.path.join(self.data_dir, 'bg_imgs/' + base_name + '.jpg'))
        img = img.convert(self.image_mode)

        # get face bbox
        bbox_path = os.path.join(self.data_dir, 'bbox/' + base_name + '.txt')

        pt2d = get_label_from_txt(bbox_path)
        x_min = pt2d[0]
        y_min = pt2d[1]
        x_max = pt2d[2]
        y_max = pt2d[3]

        # Crop the face loosely
        # k = 0 to 0.20
        k = np.random.random_sample() * 0.1
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # get pose quat
        quat_path = os.path.join(self.data_dir, 'info/' + base_name + '.txt')

        quat = get_label_from_txt(quat_path)

        # Blur?
        rnd = np.random.random_sample()
        if rnd < 0.05:
            img = img.filter(ImageFilter.BLUR)

        # Gray?
        rnd = np.random.random_sample()
        if rnd < 0.85 and base_name.find('ID') < 0:
            img = img.convert('L').convert('RGB')

        # Mask?
        rnd = np.random.random_sample()
        if rnd < 0.:
            img = add_mask(img)

        # Attention vector
        attention_vector = get_attention_vector(quat)
        vector_label = torch.FloatTensor(attention_vector)

        # classification label
        bins = np.array(range(-99, 100, self.bin_size)) / 99
        classify_label = torch.LongTensor(np.digitize(attention_vector, bins) - 1)  # 0-num_classes
        classify_label = np.where(classify_label > self.num_classes - 1, self.num_classes - 1, classify_label)
        classify_label = np.where(classify_label < 0, 0, classify_label)

        if self.transform is not None:
            img = self.transform(img)

        # RGB2BGR
        img = img[np.array([2, 1, 0]), :, :]

        return img, classify_label, vector_label, os.path.join(self.data_dir, 'bg_imgs/' + base_name + '.jpg')

    def __len__(self):
        # 168,967
        return self.length


# soft label training dataset
class TrainDataSetSORD(Dataset):
    # Head pose from 300W-LP dataset while using soft label replace one-hot label
    def __init__(self, data_dir, transform, num_classes, image_mode='RGB'):
        self.data_dir = data_dir
        self.transform = transform
        self.num_classes = num_classes
        self.bin_size = 198 // self.num_classes

        self.data_list = os.listdir(os.path.join(self.data_dir, 'bg_imgs'))

        self.image_mode = image_mode
        self.length = len(self.data_list)

    def __getitem__(self, index):
        base_name = self.data_list[index][:-4]
        img = Image.open(os.path.join(self.data_dir, 'bg_imgs/' + base_name + '.jpg'))
        img = img.convert(self.image_mode)

        # get face bbox
        bbox_path = os.path.join(self.data_dir, 'bbox/' + base_name + '.txt')

        pt2d = get_label_from_txt(bbox_path)
        x_min = pt2d[0]
        y_min = pt2d[1]
        x_max = pt2d[2]
        y_max = pt2d[3]

        # Crop the face loosely
        # k = 0 to 0.2
        k = np.random.random_sample() * 0.1
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # get pose quat
        quat_path = os.path.join(self.data_dir, 'info/' + base_name + '.txt')

        quat = get_label_from_txt(quat_path)

        # Blur?
        rnd = np.random.random_sample()
        if rnd < 0.05:
            img = img.filter(ImageFilter.BLUR)

        # Gray?
        rnd = np.random.random_sample()
        if rnd < 0.85 and base_name.find('ID') < 0:
            img = img.convert('L').convert('RGB')

        # Mask?
        rnd = np.random.random_sample()
        if rnd < 0.:
            img = add_mask(img)

        # Attention vector
        attention_vector = get_attention_vector(quat)
        vector_label = torch.FloatTensor(attention_vector)

        # classification label
        bins = np.array(range(-99, 100, self.bin_size)) / 99
        classify_label = torch.LongTensor(np.digitize(attention_vector, bins))  # 1-num_classes
        classify_label = np.where(classify_label > self.num_classes, self.num_classes, classify_label)
        classify_label = np.where(classify_label < 1, 1, classify_label)

        # soft label
        soft_label_x = get_soft_label(classify_label[0], self.num_classes)
        soft_label_y = get_soft_label(classify_label[1], self.num_classes)
        soft_label_z = get_soft_label(classify_label[2], self.num_classes)

        soft_label = torch.stack([soft_label_x, soft_label_y, soft_label_z])

        if self.transform is not None:
            img = self.transform(img)

        # RGB2BGR
        img = img[np.array([2, 1, 0]), :, :]

        return img, soft_label, vector_label, os.path.join(self.data_dir, 'bg_imgs/' + base_name + '.jpg')

    def __len__(self):
        # 168,967
        return self.length


# direct regression testing dataset
class TestDataSetReg(Dataset):
    def __init__(self, data_dir, transform, image_mode='RGB', length=None):
        self.data_dir = data_dir
        self.transform = transform

        self.data_list = os.listdir(os.path.join(self.data_dir, 'bg_imgs'))

        self.image_mode = image_mode
        if length is None:
            self.length = len(self.data_list)
        else:
            self.length = length

    def __getitem__(self, index):
        base_name = self.data_list[index][:-4]
        img = Image.open(os.path.join(self.data_dir, 'bg_imgs/' + base_name + '.jpg'))
        img = img.convert(self.image_mode)

        # get face bbox
        bbox_path = os.path.join(self.data_dir, 'bbox/' + base_name + '.txt')

        pt2d = get_label_from_txt(bbox_path)
        x_min = pt2d[0]
        y_min = pt2d[1]
        x_max = pt2d[2]
        y_max = pt2d[3]

        # Crop the face loosely
        # k = 0.2
        k = 0.1
        x_min -= k * abs(x_max - x_min)
        y_min -= k * abs(y_max - y_min)
        x_max += k * abs(x_max - x_min)
        y_max += 0.3 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # get pose angle pitch,yaw,roll(degrees)
        angle_path = os.path.join(self.data_dir, 'angles/' + base_name + '.txt')
        if os.path.exists(angle_path):
            angle = get_label_from_txt(angle_path)
            angle = torch.FloatTensor(angle)
        else:
            angle = 30.0

        # get pose quat
        quat_path = os.path.join(self.data_dir, 'info/' + base_name + '.txt')
        quat = get_label_from_txt(quat_path)

        # Attention vector
        attention_vector = get_attention_vector(quat)
        vector_label = torch.FloatTensor(attention_vector)

        if self.transform is not None:
            img = self.transform(img)

        # RGB2BGR
        img = img[np.array([2, 1, 0]), :, :]

        return img, vector_label, angle, torch.FloatTensor(pt2d), os.path.join(self.data_dir, 'bg_imgs/' + base_name + '.jpg')

    def __len__(self):
        # 1,969
        return self.length


# classify and regression testing dataset
class TestDataSetCls(Dataset):
    def __init__(self, data_dir, transform, num_classes, image_mode='RGB', length=None):
        self.data_dir = data_dir
        self.transform = transform
        self.num_classes = num_classes
        self.bin_size = 198 // self.num_classes

        self.data_list = os.listdir(os.path.join(self.data_dir, 'bg_imgs'))

        self.image_mode = image_mode
        if length is None:
            self.length = len(self.data_list)
        else:
            self.length = length

    def __getitem__(self, index):
        base_name = self.data_list[index][:-4]
        img = Image.open(os.path.join(self.data_dir, 'bg_imgs/' + base_name + '.jpg'))
        img = img.convert(self.image_mode)

        # get face bbox
        bbox_path = os.path.join(self.data_dir, 'bbox/' + base_name + '.txt')

        pt2d = get_label_from_txt(bbox_path)
        x_min = pt2d[0]
        y_min = pt2d[1]
        x_max = pt2d[2]
        y_max = pt2d[3]

        # Crop the face loosely
        k = 0.1
        x_min -= k * abs(x_max - x_min)
        y_min -= k * abs(y_max - y_min)
        x_max += k * abs(x_max - x_min)
        y_max += 0.3 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # get pose angle pitch,yaw,roll(degrees)
        angle_path = os.path.join(self.data_dir, 'angles/' + base_name + '.txt')
        angle = get_label_from_txt(angle_path)
        angle = torch.FloatTensor(angle)

        # get pose quat
        quat_path = os.path.join(self.data_dir, 'info/' + base_name + '.txt')
        quat = get_label_from_txt(quat_path)

        # Attention vector
        attention_vector = get_attention_vector(quat)
        vector_label = torch.FloatTensor(attention_vector)

        # classification label
        bins = np.array(range(-99, 100, self.bin_size)) / 99
        classify_label = torch.LongTensor(np.digitize(attention_vector, bins) - 1)  # 0-num_classes
        classify_label = np.where(classify_label > self.num_classes - 1, self.num_classes - 1, classify_label)
        classify_label = np.where(classify_label < 0, 0, classify_label)

        if self.transform is not None:
            img = self.transform(img)

        # RGB2BGR
        img = img[np.array([2, 1, 0]), :, :]

        return img, classify_label, vector_label, angle, torch.FloatTensor(pt2d), os.path.join(self.data_dir, 'bg_imgs/' + base_name + '.jpg')

    def __len__(self):
        # 1,969
        return self.length


# soft label testing dataset
class TestDataSetSORD(Dataset):
    def __init__(self, data_dir, transform, num_classes, image_mode='RGB', length=None):
        self.data_dir = data_dir
        self.transform = transform
        self.num_classes = num_classes
        self.bin_size = 198 // self.num_classes

        self.data_list = os.listdir(os.path.join(self.data_dir, 'bg_imgs'))

        self.image_mode = image_mode
        if length is None:
            self.length = len(self.data_list)
        else:
            self.length = length

    def __getitem__(self, index):
        base_name = self.data_list[index][:-4]
        img = Image.open(os.path.join(self.data_dir, 'bg_imgs/' + base_name + '.jpg'))
        img = img.convert(self.image_mode)

        # get face bbox
        bbox_path = os.path.join(self.data_dir, 'bbox/' + base_name + '.txt')

        pt2d = get_label_from_txt(bbox_path)
        x_min = pt2d[0]
        y_min = pt2d[1]
        x_max = pt2d[2]
        y_max = pt2d[3]

        # Crop the face loosely
        k = 0.1
        x_min -= k * abs(x_max - x_min)
        y_min -= k * abs(y_max - y_min)
        x_max += k * abs(x_max - x_min)
        y_max += 0.3 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # get pose angle pitch,yaw,roll(degrees)
        angle_path = os.path.join(self.data_dir, 'angles/' + base_name + '.txt')
        angle = get_label_from_txt(angle_path)
        angle = torch.FloatTensor(angle)

        # get pose quat
        quat_path = os.path.join(self.data_dir, 'info/' + base_name + '.txt')
        quat = get_label_from_txt(quat_path)

        # Attention vector
        attention_vector = get_attention_vector(quat)
        vector_label = torch.FloatTensor(attention_vector)

        # classification label
        bins = np.array(range(-99, 100, self.bin_size)) / 99
        classify_label = torch.LongTensor(np.digitize(attention_vector, bins))  # 1-num_classes
        classify_label = np.where(classify_label > self.num_classes, self.num_classes, classify_label)
        classify_label = np.where(classify_label < 1, 1, classify_label)

        # soft label
        soft_label_x = get_soft_label(classify_label[0], self.num_classes)
        soft_label_y = get_soft_label(classify_label[1], self.num_classes)
        soft_label_z = get_soft_label(classify_label[2], self.num_classes)

        soft_label = torch.stack([soft_label_x, soft_label_y, soft_label_z])

        if self.transform is not None:
            img = self.transform(img)

        # RGB2BGR
        img = img[np.array([2, 1, 0]), :, :]

        return img, soft_label, vector_label, angle, torch.FloatTensor(pt2d), os.path.join(self.data_dir, 'bg_imgs/' + base_name + '.jpg')

    def __len__(self):
        # 1,969
        return self.length
