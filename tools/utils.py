import os
import cv2
import tqdm
import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from PIL import Image
from thop import profile
from math import cos, sin
from tools.rotation import Rotation as R
from tools.flops_benchmark import add_flops_counting_methods


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


def softmax_temperature(tensor, temperature):
    result = torch.exp(tensor / temperature)
    result = torch.div(result, torch.sum(result, 1).unsqueeze(1).expand_as(result))
    return result


def get_pose_params_from_mat(mat_path):
    # This functions gets the pose parameters from the .mat
    # Annotations that come with the Pose_300W_LP dataset.
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll, tdx, tdy]
    pose_params = pre_pose_params[:5]
    return pose_params


def get_ypr_from_mat(mat_path):
    # Get yaw, pitch, roll from .mat annotation.
    # They are in radians
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose_params = pre_pose_params[:3]
    return pose_params


def get_label_from_txt(txt_path):
    with open(txt_path, 'r') as fr:
        line = fr.read().splitlines()
    line = line[0].split(' ')
    label = [float(i) for i in line]
    return label


def get_pt2d_from_mat(mat_path):
    # Get 2D landmarks
    mat = sio.loadmat(mat_path)
    pt2d = mat['pt2d']
    return pt2d


def mse_loss(input, target):
    return torch.sum(torch.abs(input.data - target.data) ** 2)


def plot_pose_cube(img, yaw, pitch, roll, tdx=None, tdy=None, size=150.):
    # Input is a cv2 image
    # pose_params: (pitch, yaw, roll, tdx, tdy)
    # Where (tdx, tdy) is the translation of the face.
    # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]

    p = pitch * np.pi / 180
    y = -(yaw * np.pi / 180)
    r = roll * np.pi / 180
    if tdx != None and tdy != None:
        face_x = tdx - 0.50 * size
        face_y = tdy - 0.50 * size
    else:
        height, width = img.shape[:2]
        face_x = width / 2 - 0.5 * size
        face_y = height / 2 - 0.5 * size

    x1 = size * (cos(y) * cos(r)) + face_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y
    x2 = size * (-cos(y) * sin(r)) + face_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
    x3 = size * (sin(y)) + face_x
    y3 = size * (-cos(y) * sin(p)) + face_y

    # Draw base in red
    cv2.line(img, (int(face_x), int(face_y)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(img, (int(face_x), int(face_y)), (int(x2), int(y2)), (0, 0, 255), 3)
    cv2.line(img, (int(x2), int(y2)), (int(x2 + x1 - face_x), int(y2 + y1 - face_y)), (0, 0, 255), 3)
    cv2.line(img, (int(x1), int(y1)), (int(x1 + x2 - face_x), int(y1 + y2 - face_y)), (0, 0, 255), 3)
    # Draw pillars in blue
    cv2.line(img, (int(face_x), int(face_y)), (int(x3), int(y3)), (255, 0, 0), 2)
    cv2.line(img, (int(x1), int(y1)), (int(x1 + x3 - face_x), int(y1 + y3 - face_y)), (255, 0, 0), 2)
    cv2.line(img, (int(x2), int(y2)), (int(x2 + x3 - face_x), int(y2 + y3 - face_y)), (255, 0, 0), 2)
    cv2.line(img, (int(x2 + x1 - face_x), int(y2 + y1 - face_y)),
             (int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)), (255, 0, 0), 2)
    # Draw top in green
    cv2.line(img, (int(x3 + x1 - face_x), int(y3 + y1 - face_y)),
             (int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)), (0, 255, 0), 2)
    cv2.line(img, (int(x2 + x3 - face_x), int(y2 + y3 - face_y)),
             (int(x3 + x1 + x2 - 2 * face_x), int(y3 + y2 + y1 - 2 * face_y)), (0, 255, 0), 2)
    cv2.line(img, (int(x3), int(y3)), (int(x3 + x1 - face_x), int(y3 + y1 - face_y)), (0, 255, 0), 2)
    cv2.line(img, (int(x3), int(y3)), (int(x3 + x2 - face_x), int(y3 + y2 - face_y)), (0, 255, 0), 2)

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


def degress_score(cos_value, error_degrees):
    """
    get collect score
    :param cos_value: cos value of two vectors
    :param error_degrees: degrees error limit value,integer
    :return:
    """
    score = torch.tensor([1.0 if i > cos(error_degrees * np.pi / 180) else 0.0 for i in cos_value])
    return score


def norm_vector(v):
    """
    normalization vector
    :param v: vector
    :return:
    """
    vector_len = v.norm(dim=-1)
    v = v / vector_len.unsqueeze(dim=-1)
    return v


def l1_norm(x):
    """
    l1 normalization
    :param x:
    :return:
    """
    x = torch.div(x, torch.sum(x, dim=-1))
    return x


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

    v_top = np.mat([[0], [1], [0]])
    v_top = dcm * v_top
    v_top = np.array(v_top).reshape(3)

    return np.hstack([v_front, v_top])


def get_flops(net, input_size=(1, 3, 224, 224), method='benchmark'):
    """
    compute model flops
    :param net: model with weight initialized
    :param input_size:
    :param method:
    :return:
    """
    if method == 'profile':
        flops, param = profile(net, input_size)
        flops = flops / 1e6
    else:
        inputs = torch.randn(input_size)

        net = add_flops_counting_methods(net)
        net = net.train()
        net.start_flops_count()

        _ = net(inputs)
        flops = net.compute_average_flops_cost() / 1e6 / 2
    return flops


def add_mask(img):
    """
    augmentation: add mask
    :param img:
    :return:
    """
    w, h = img.size

    min_w = w * np.random.randint(1000, 1125) * 0.0001
    max_w = w * np.random.randint(1600, 1800) * 0.0001
    min_h = h * np.random.randint(1000, 1125) * 0.0001
    max_h = h * np.random.randint(1600, 1800) * 0.0001

    mask_w = np.random.randint(min_w, max_w)
    mask_h = np.random.randint(min_h, max_h)

    if (mask_w * mask_h) < (w * h * 0.1):
        start_w = np.random.randint(0, w - mask_w)
        start_h = np.random.randint(0, h - mask_h)
        mask = np.zeros((mask_h, mask_w, 3))
        img = np.array(img)
        img[start_h:start_h + mask_h, start_w:start_w + mask_w, :] = mask
        img = Image.fromarray(img)
    return img


def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def vector2pyr(v_front, v_top):
    """
    calc angles between vector and three plane
    :param v_front:face orientation vector
    :param v_top:face right-part face orientation vector
    :return:degrees of pitch,yaw and roll,but roll not useful
    """
    yaw = np.arcsin(vector_cos(np.array([1, 0, 0]), v_front)) * 180 / np.pi
    pitch = np.arcsin(vector_cos(np.array([0, 1, 0]), v_front)) * 180 / np.pi
    roll = np.arcsin(vector_cos(np.array([1, 0, 0]), v_top)) * 180 / np.pi

    return np.array([pitch, yaw, roll])


def classify2vector(x, y, z, softmax, num_classes, top_k, cls2reg):
    """
    get vector from classify results
    :param x: fc_x output,np.ndarray(66,)
    :param y: fc_y output,np.ndarray(66,)
    :param z: fc_z output,np.ndarray(66,)
    :param softmax: softmax function
    :param num_classes: number of classify, integer
    :param top_k: using top K expectation replace whole expectation,integer
    :param cls2reg: method for compute regression value from classify results,[Average,Expectation]
    :return: 
    """
    idx_tensor = [idx for idx in range(num_classes)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(0)

    x_probability = softmax(x)
    y_probability = softmax(y)
    z_probability = softmax(z)

    if top_k is not None:
        x_topk_id = torch.argsort(x_probability, descending=True)[:, :top_k]
        y_topk_id = torch.argsort(y_probability, descending=True)[:, :top_k]
        z_topk_id = torch.argsort(z_probability, descending=True)[:, :top_k]
        if cls2reg == 'Average':
            x_pred = torch.mean(x_topk_id.float(), dim=-1) * (198 // num_classes) - 96
            y_pred = torch.mean(y_topk_id.float(), dim=-1) * (198 // num_classes) - 96
            z_pred = torch.mean(z_topk_id.float(), dim=-1) * (198 // num_classes) - 96
        elif cls2reg == 'Expectation':
            # get top_k probability
            x_topk_probability = torch.stack([x_probability[i][x_topk_id[i]] for i in range(x_probability.size(0))])
            y_topk_probability = torch.stack([y_probability[i][y_topk_id[i]] for i in range(y_probability.size(0))])
            z_topk_probability = torch.stack([z_probability[i][z_topk_id[i]] for i in range(z_probability.size(0))])

            # normalize top_k probability
            x_topk_probability = x_topk_probability / torch.sum(x_topk_probability, dim=-1).unsqueeze(dim=-1)
            y_topk_probability = y_topk_probability / torch.sum(y_topk_probability, dim=-1).unsqueeze(dim=-1)
            z_topk_probability = z_topk_probability / torch.sum(z_topk_probability, dim=-1).unsqueeze(dim=-1)

            x_pred = torch.sum(x_topk_probability * x_topk_id.float(), dim=-1) * (198 // num_classes) - 96
            y_pred = torch.sum(y_topk_probability * y_topk_id.float(), dim=-1) * (198 // num_classes) - 96
            z_pred = torch.sum(z_topk_probability * z_topk_id.float(), dim=-1) * (198 // num_classes) - 96
    else:
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
