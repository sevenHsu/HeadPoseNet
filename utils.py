# -*- coding:utf-8 -*-
"""
    utils script
"""
import os
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import cos, sin
from rotation import Rotation as R


def mkdir(dir_path):
    """
    build directory
    :param dir_path:
    :return:
    """
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def norm_vector(v):
    """
    normalization vector
    :param v: vector
    :return:
    """
    vector_len = v.norm(dim=-1)
    v = v / vector_len.unsqueeze(dim=-1)

    return v


def vector_cos(u, v):
    """
    compute cos value between two vectors
    :param u:
    :param v:
    :return:
    """
    assert u.shape == v.shape, 'shape of two vectors should be same'
    cos_value = torch.sum(u * v, dim=1) / torch.sqrt(torch.sum(u ** 2, dim=1) * torch.sum(v ** 2, dim=1))

    return cos_value


def load_filtered_stat_dict(model, snapshot):
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


def softmax(input):
    """
    implementation of softmax with numpy
    :param input:
    :return:
    """
    input = input - np.max(input)
    input_exp = np.exp(input)
    input_exp_sum = np.sum(input_exp)

    return input_exp / input_exp_sum + (10 ** -6)


def draw_bbox(img, bbox):
    """
    draw face bounding box
    :param img:np.ndarray(H,W,C)
    :param bbox: list[x1,y1,x2,y2]
    :return:
    """
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2])
    y2 = int(bbox[3])

    img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255))
    return img


def draw_front(img, x, y, tdx=None, tdy=None, size=100, color=(0, 255, 0)):
    """
    draw face orientation vector in image
    :param img: face image
    :param x: x of face orientation vector,integer
    :param y: y of face orientation vector,integer
    :param tdx: x of start point,integer
    :param tdy: y of start point,integer
    :param size: length of face orientation vector
    :param color:
    :return:
    """
    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2
    x2 = tdx - size * x
    y2 = tdy - size * y
    cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x2), int(y2)), color, 2, tipLength=0.3)
    return img


def draw_axis(img, pitch, yaw, roll, tdx=None, tdy=None, size=100):
    """
    :param img: original images.[np.ndarray]
    :param yaw:
    :param pitch:
    :param roll:
    :param tdx: x-axis for start point
    :param tdy: y-axis for start point
    :param size: line size
    :return:
    """
    pitch = pitch
    yaw = -yaw
    roll = roll

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 255, 255), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

    return img


def get_label_from_txt(txt_path):
    with open(txt_path, 'r') as fr:
        line = fr.read().splitlines()
    line = line[0].split(' ')
    label = [float(i) for i in line]

    return label


def degress_score(cos_value, error_degrees):
    """
    get collect score
    :param cos_value: cos value of two vectors
    :param error_degrees: degrees error limit value,integer
    :return:
    """
    score = torch.tensor([1.0 if i > cos(error_degrees * np.pi / 180) else 0.0 for i in cos_value])
    return score


def get_attention_vector(quat):
    """
    get face orientation vector from quaternion
    :param quat:
    :return:
    """
    dcm = R.quat2dcm(quat)
    v_front = np.mat([[0], [0], [1]])
    v_front = dcm * v_front
    v_front = np.array(v_front).reshape(3)

    # v_top = np.mat([[0], [1], [0]])
    # v_top = dcm * v_top
    # v_top = np.array(v_top).reshape(3)

    # return np.hstack([v_front, v_top])
    return v_front


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


def computeLoss(classify_label, vector_label, logits, softmax, cls_criterion, reg_criterion, args):
    # get x,y,z cls label
    x_cls_label = classify_label[:, 0]
    y_cls_label = classify_label[:, 1]
    z_cls_label = classify_label[:, 2]

    # get x,y,z continue label
    x_reg_label = vector_label[:, 0]
    y_reg_label = vector_label[:, 1]
    z_reg_label = vector_label[:, 2]

    x_cls_pred, y_cls_pred, z_cls_pred = logits

    # CrossEntry loss(for classify)
    x_cls_loss = cls_criterion(x_cls_pred, x_cls_label)
    y_cls_loss = cls_criterion(y_cls_pred, y_cls_label)
    z_cls_loss = cls_criterion(z_cls_pred, z_cls_label)

    # get prediction vector(get continue value from classify result)
    x_reg_pred, y_reg_pred, z_reg_pred, vector_pred = classify2vector(x_cls_pred, y_cls_pred, z_cls_pred, softmax, args.num_classes)

    # Regression loss
    x_reg_loss = reg_criterion(x_reg_pred, x_reg_label)
    y_reg_loss = reg_criterion(y_reg_pred, y_reg_label)
    z_reg_loss = reg_criterion(z_reg_pred, z_reg_label)

    # Total loss
    x_loss = x_cls_loss + args.alpha * x_reg_loss
    y_loss = y_cls_loss + args.alpha * y_reg_loss
    z_loss = z_cls_loss + args.alpha * z_reg_loss

    loss = [x_loss, y_loss, z_loss]

    # get degree error
    cos_value = vector_cos(vector_pred, vector_label)
    degree_error = torch.mean(torch.acos(cos_value) * 180 / np.pi)

    return loss, degree_error


def classify2vector(x, y, z, softmax, num_classes):
    """
    get vector from classify results
    :param x: fc_x output,np.ndarray(66,)
    :param y: fc_y output,np.ndarray(66,)
    :param z: fc_z output,np.ndarray(66,)
    :param softmax: softmax function
    :param num_classes: number of classify, integer
    :return:
    """
    idx_tensor = [idx for idx in range(num_classes)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(0)

    x_probability = softmax(x)
    y_probability = softmax(y)
    z_probability = softmax(z)

    x_pred = torch.sum(x_probability * idx_tensor, dim=-1) * (198 // num_classes) - 96
    y_pred = torch.sum(y_probability * idx_tensor, dim=-1) * (198 // num_classes) - 96
    z_pred = torch.sum(z_probability * idx_tensor, dim=-1) * (198 // num_classes) - 96

    pred_vector = torch.stack([x_pred, y_pred, z_pred]).transpose(1, 0)
    pred_vector = norm_vector(pred_vector)

    # split to x,y,z
    x_reg = pred_vector[:, 0]
    y_reg = pred_vector[:, 1]
    z_reg = pred_vector[:, 2]

    return x_reg, y_reg, z_reg, pred_vector


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
        plt.subplot(1, 3, i + 1)
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


def collect_score(degree_dict, save_dir):
    """

    :param save_dir:
    :return:
    """
    plt.switch_backend('agg')
    x = np.array(range(0, 181, 5))
    degree_error = degree_dict['degree_error']
    mount = np.zeros(len(x))
    for j in range(len(x)):
        mount[j] = sum(degree_error < x[j])
    y = mount / len(degree_error)
    plt.plot(x, y, c="red", label="MobileNetV2")
    plt.legend(loc='lower right', fontsize='x-small')
    plt.xlabel('degrees upper limit')
    plt.ylabel('accuracy')
    plt.xlim(0, 105)
    plt.ylim(0., 1.05)
    plt.xticks([j for j in range(0, 105, 5)], [j for j in range(0, 105, 5)])
    plt.yticks([j / 100 for j in range(0, 105, 5)], [j / 100 for j in range(0, 105, 5)])
    plt.title("accuracy under degree upper limit")
    plt.grid()
    plt.savefig(save_dir + '/collect_score.png')
