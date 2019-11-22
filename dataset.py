# -*-coding:utf-8 -*-
"""
    DataSet class
"""
import os
import torch
import numpy as np
from PIL import Image
from PIL import ImageFilter
from utils import get_soft_label
from torchvision import transforms
from utils import get_label_from_txt
from torch.utils.data import DataLoader
from utils import get_attention_vector
from torch.utils.data.dataset import Dataset


def loadData(data_dir, input_size, batch_size, num_classes, training=True):
    """

    :return:
    """
    # define transformation
    if training:
        transformations = transforms.Compose([transforms.Resize(int(np.ceil(input_size * 1.0714))),
                                              transforms.RandomCrop(input_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        dataset = TrainDataSet(data_dir, transformations, num_classes)
        train_data_length = int(dataset.length * 0.8)
        valid_data_length = dataset.length - train_data_length
        train_data, valid_data = torch.utils.data.random_split(dataset, [train_data_length, valid_data_length])
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=4)
        return train_loader, valid_loader
    else:
        transformations = transforms.Compose([transforms.Resize(input_size),
                                              transforms.RandomCrop(input_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        test_dataset = TestDataSet(data_dir, transformations, num_classes)

        # initialize train DataLoader
        data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

        return data_loader


class TrainDataSet(Dataset):
    def __init__(self, data_dir, transform, num_classes, image_mode="RGB"):
        self.data_dir = data_dir
        self.transform = transform
        self.num_classes = num_classes
        self.bin_size = 198 // self.num_classes
        self.image_mode = image_mode
        self.bins = np.array(range(-99, 100, self.bin_size)) / 99

        self.data_list = os.listdir(os.path.join(self.data_dir, 'bg_imgs'))
        self.length = len(self.data_list)

    def __getitem__(self, index):
        # data basename
        base_name, _ = self.data_list[index].split('.')

        # read image file
        img = Image.open(os.path.join(self.data_dir, "bg_imgs/" + base_name + ".jpg"))
        img = img.convert(self.image_mode)

        # get face bounding box
        pt2d = get_label_from_txt(os.path.join(self.data_dir, "bbox/" + base_name + ".txt"))
        x_min, y_min, x_max, y_max = pt2d

        # crop face loosely:k=0to 0.2
        k = np.random.random_sample() * 0.1
        x_min -= 0.6 * k * abs(x_max - x_min)
        y_min -= k * abs(y_max - y_min)
        x_max += 0.6 * k * abs(x_max - x_min)
        y_max += 0.6 * k * abs(y_max - y_min)
        img = img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))

        # Augmentation:Blur?
        if np.random.random_sample() < 0.05:
            img = img.filter(ImageFilter.BLUR)

        # Augmentation:Gray?
        if np.random.random_sample() < 0.5 and base_name.find('ID') < 0:
            img = img.convert('L').convert("RGB")

        # transform
        if self.transform:
            img = self.transform(img)

        # RGB2BGR
        img = img[np.array([2, 1, 0]), :, :]

        # get pose quat
        quat = get_label_from_txt(os.path.join(self.data_dir, "info/" + base_name + '.txt'))

        # face orientation vector
        vector_label = get_attention_vector(quat)
        vector_label = torch.FloatTensor(vector_label)

        # classification label
        classify_label = torch.LongTensor(np.digitize(vector_label, self.bins))
        classify_label = np.where(classify_label > self.num_classes, self.num_classes, classify_label)
        classify_label = np.where(classify_label < 1, 1, classify_label)

        # soft label
        soft_label_x = get_soft_label(classify_label[0], self.num_classes)
        soft_label_y = get_soft_label(classify_label[1], self.num_classes)
        soft_label_z = get_soft_label(classify_label[2], self.num_classes)

        soft_label = torch.stack([soft_label_x, soft_label_y, soft_label_z])

        return img, soft_label, vector_label, os.path.join(self.data_dir, "bg_imgs/" + base_name + ".jpg")

    def __len__(self):
        return self.length


class TestDataSet(Dataset):
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
