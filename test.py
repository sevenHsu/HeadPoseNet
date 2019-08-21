# -*- coding:utf-8 -*-
import os
import tqdm
import json
import torch
import argparse
import cv2 as cv
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tools import *
from torchvision import transforms
from torch.utils.data import DataLoader


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--test_data', dest='test_data', help='Directory path for data.',
                        default='/home/AFLW2000QUAT', type=str)
    parser.add_argument('--snapshot', dest='snapshot', help='Name of model snapshot.',
                        default='/train/trainset/1/results/shufflenet/snapshots/shuffle_net_epoch_25.pkl', type=str)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
                        default=64, type=int)
    parser.add_argument('--degree_error_limit', dest='degree_error_limit', help='degrees error for calc cs',
                        default=10, type=int)
    parser.add_argument('--save_dir', dest='save_dir', help='directory for saving drawn pic',
                        default='/train/trainset/1/results/shufflenet', type=str)
    parser.add_argument('--show_front', dest='show_front', help='show front or not',
                        default=False, type=bool)
    parser.add_argument('--analysis', dest='analysis', help='analysis result or not',
                        default=False, type=bool)
    parser.add_argument('--huge_error', dest='huge_error', help='show huge error or not',
                        default=False, type=bool)
    parser.add_argument('--collect_score', dest='collect_score', help='show huge error or not',
                        default=False, type=bool)
    parser.add_argument('--net', dest='net', help='net name for training', choices=['resnet', 'mobilenet', 'shufflenet', 'squeezenet'],
                        default='net', type=str)
    parser.add_argument('--mode', dest='mode', help='direct regression or cls or sord', choices=['reg', 'cls', 'sord'],
                        default='reg', type=str)
    parser.add_argument('--top_k', dest='top_k', help='top_k for cls2reg',
                        default=None, type=int)
    parser.add_argument('--num_classes', dest='num_classes', help='number of classify',
                        default=33, type=int)
    parser.add_argument('--cls2reg', dest='cls2reg', choices=['Expectation', 'Average'], help='number of classify',
                        default='Average', type=str)
    parser.add_argument('--width_mult', dest='width_mult', choices=[0.5, 1.0], help='mobilenet_v2 width_mult',
                        default=1.0, type=float)
    parser.add_argument('--input_size', dest='input_size', choices=[224, 192, 160, 128, 96], help='size of input images',
                        default=224, type=int)
    parser.add_argument('--squeeze_version', dest='squeeze_version', choices=['1_0', '1_1'], help='size of input images',
                        default='1_1', type=str)

    args = parser.parse_args()
    return args


def draw_attention_vector(vector_label, angle_label, pred_vector, img_path, pt2d, args):
    save_dir = os.path.join(args.save_dir, 'show_front')
    img_name = os.path.basename(img_path)

    img = cv.imread(img_path)

    predx, predy, predz = pred_vector

    start_x = (pt2d[0].item() + pt2d[2].item()) // 2
    start_y = (pt2d[1].item() + pt2d[3].item()) // 2

    # draw GT attention vector with green
    if 'DMS_TEST_DATA' in args.test_data.split('/'):
        gtx, gty, gtz = vector_label
        draw_front(img, gtx, gty, tdx=start_x, tdy=start_y, size=100, color=(0, 255, 0))

    # draw GT axis
    elif 'AFLW2000QUAT' in args.test_data.split('/'):
        pitch, yaw, roll = angle_label
        draw_axis(img, pitch, yaw, roll, tdx=start_x, tdy=start_y, size=100)

    # draw face bbox
    draw_bbox(img, pt2d)

    # draw pred attention vector with red
    draw_front(img, predx, predy, tdx=start_x, tdy=start_y, size=100, color=(0, 0, 255))

    cv.imwrite(os.path.join(save_dir, img_name), img)


def draw_huge_error(loss_dict, args):
    degrees_error = loss_dict['degree_error']
    img_names = loss_dict['img_name']
    nums = min(100, sum(np.where(np.array(degrees_error) > args.degree_error_limit, 1, 0)))
    rows = int(np.sqrt(nums))
    cols = int(np.ceil(nums / rows))

    idx = np.argsort(-1 * np.array(degrees_error))[:nums]
    huge_error_img_list = np.array(img_names)[idx]
    huge_error_degrees = np.array(degrees_error)[idx]

    img_dir = os.path.join(args.save_dir, 'show_front')
    huge_error_dir = os.path.join(args.save_dir, 'huge_error')
    all_img = np.zeros((rows * 300, cols * 300, 3))
    for k, huge_error_img in enumerate(huge_error_img_list):
        i = k // rows
        j = k - (i * rows)
        img_name = os.path.basename(huge_error_img)
        img = cv.imread(os.path.join(img_dir, img_name))
        degree_error = huge_error_degrees[k]
        img[2:30, 10:330, :] = 0
        cv.putText(img, str(degree_error), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        img = cv.resize(img, (300, 300))
        all_img[i * 300:(i + 1) * 300, j * 300:(j + 1) * 300, :] = img
    cv.imwrite(huge_error_dir + '/huge_error.jpg', all_img)


def test_reg(model, test_loader, args):
    if args.analysis:
        mkdir(os.path.join(args.save_dir, 'analysis'))
        loss_dict = {'img_name': list(), 'angles': list(), 'degree_error': list()}
    error = 0.0
    total = 0.0
    score = 0.0
    for i, (images, vector_label, angle_label, pt2d, names) in enumerate(tqdm.tqdm(test_loader)):
        with torch.no_grad():
            images = images.cuda(0)
            vector_label = vector_label.cuda(0)

            pred_vector = model(images)
            pred_vector = norm_vector(pred_vector)

            # Mean absolute error
            cos_value = vector_cos(pred_vector, vector_label)
            degrees_error = torch.acos(cos_value) * 180 / np.pi

            # save euler angle and degrees error to loss_dict
            if args.analysis:
                for k in range(len(angle_label)):
                    loss_dict['img_name'].append(names[k])
                    loss_dict['angles'].append(angle_label[k].tolist())  # pitch,yaw,roll
                    loss_dict['degree_error'].append(float(degrees_error[k]))

            # collect error
            error += torch.sum(degrees_error)
            score += torch.sum(degress_score(cos_value, args.degree_error_limit))

            total += vector_label.size(0)

            # Save first image in batch with pose cube or axis.
            if args.show_front:
                mkdir(os.path.join(args.save_dir, 'show_front'))
                for j in range(vector_label.size(0)):
                    draw_attention_vector(vector_label[j].cpu().tolist(),
                                          angle_label[j].cpu().tolist(),
                                          pred_vector[j].cpu().tolist(),
                                          names[j],
                                          pt2d[j],
                                          args)

    avg_error = error / total
    total_score = score / total
    print('Average degree Error:%.4f | score with error<10º:%.4f' % (avg_error.item(), total_score.item()))

    # save analysis of loss distribute
    if args.analysis:
        print('analysis result')
        analysis_tools.show_loss_distribute(loss_dict, os.path.join(args.save_dir, 'analysis'), os.path.basename(snapshot_path).split('.')[0])

    # save collect score curve
    if args.collect_score:
        print("saving degrees error dict")
        with open(os.path.join(args.save_dir, 'collect_score/' + os.path.basename(snapshot_path).split('.')[0] + '.json'), 'w') as fw:
            json.dump(loss_dict, fw)

    # draw huge error images
    if args.huge_error:
        print("drawing huge error images......")
        draw_huge_error(loss_dict, args)


def test_cls(model, test_loader, softmax, args):
    if args.analysis:
        mkdir(os.path.join(args.save_dir, 'analysis'))
        loss_dict = {'img_name': list(), 'angles': list(), 'degree_error': list()}
    error = 0.0
    total = 0.0
    score = 0.0
    for i, (images, classify_label, vector_label, angle_label, pt2d, names) in enumerate(tqdm.tqdm(test_loader)):
        with torch.no_grad():
            images = images.cuda(0)
            vector_label = vector_label.cuda(0)

            # get x,y,z cls predictions
            x_cls_pred, y_cls_pred, z_cls_pred = model(images)

            # get prediction vector(get continue value from classify result)
            _, _, _, pred_vector = classify2vector(x_cls_pred, y_cls_pred, z_cls_pred, softmax,
                                                   num_classes=args.num_classes,
                                                   top_k=args.top_k, cls2reg=args.cls2reg)

            # Mean absolute error
            cos_value = vector_cos(pred_vector, vector_label)
            degrees_error = torch.acos(cos_value) * 180 / np.pi

            # save euler angle and degrees error to loss_dict
            if args.analysis:
                for k in range(len(angle_label)):
                    loss_dict['img_name'].append(names[k])
                    loss_dict['angles'].append(angle_label[k].tolist())  # pitch,yaw,roll
                    loss_dict['degree_error'].append(float(degrees_error[k]))

            # collect error
            error += torch.sum(degrees_error)
            score += torch.sum(degress_score(cos_value, args.degree_error_limit))

            total += vector_label.size(0)

            # Save first image in batch with pose cube or axis.
            if args.show_front:
                mkdir(os.path.join(args.save_dir, 'show_front'))
                for j in range(vector_label.size(0)):
                    draw_attention_vector(vector_label[j].cpu().tolist(),
                                          angle_label[j].cpu().tolist(),
                                          pred_vector[j].cpu().tolist(),
                                          names[j],
                                          pt2d[j],
                                          args)

    avg_error = error / total
    total_score = score / total
    print('Average degree Error:%.4f | score with error<10º:%.4f' % (avg_error.item(), total_score.item()))

    # save analysis of loss distribute
    if args.analysis:
        print('analysis result')
        show_loss_distribute(loss_dict, os.path.join(args.save_dir, 'analysis'), os.path.basename(snapshot_path).split('.')[0])

    # save collect score curve
    if args.collect_score:
        print("saving degrees error dict")
        with open(os.path.join(args.save_dir, 'collect_score/' + os.path.basename(snapshot_path).split('.')[0] + '.json'), 'w') as fw:
            json.dump(loss_dict, fw)

    # draw huge error images
    if args.huge_error:
        print("drawing huge error images......")
        draw_huge_error(loss_dict, args)


if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    snapshot_path = args.snapshot
    mkdir(args.save_dir)

    # cls and sord
    print("Loading model weight......")
    if args.mode in ['cls', 'sord']:
        if args.net == 'resnet':
            model = ResNetCls(layers=[3, 4, 6, 3], num_classes=args.num_classes)
        elif args.net == 'mobilenet':
            model = MobileNetV2Cls(num_classes=args.num_classes, width_mult=args.width_mult)
        elif args.net == 'shufflenet':
            model = ShuffleNetV2Cls(num_classes=args.num_classes, scale=args.width_mult)
        elif args.net == 'squeezenet':
            model = SqueezeNetCls(num_classes=args.num_classes)
    # reg
    elif args.mode == 'reg':
        if args.net == 'resnet':
            model = ResNetReg(layers=[3, 4, 6, 3], num_classes=3)
        elif args.net == 'mobilenet':
            model = MobileNetV2Reg(num_classes=3, width_mult=args.width_mult)
        elif args.net == 'shufflenet':
            model = ShuffleNetV2Reg(num_classes=3, scale=args.width_mult)
        elif args.net == 'squeezenet':
            model = SqueezeNetCls(num_classes=3)
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)
    model.cuda(0)

    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    softmax = nn.Softmax(dim=1).cuda(0)

    # initialize transformation
    transformations = transforms.Compose([transforms.Resize(args.input_size),
                                          transforms.CenterCrop(args.input_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # load test data
    print("Loading test data......")
    if args.mode == 'reg':
        test_dataset = TestDataSetReg(args.test_data, transformations)
    elif args.mode == 'cls':
        test_dataset = TestDataSetCls(args.test_data, transformations, args.num_classes)
    elif args.mode == 'sord':
        test_dataset = TestDataSetSORD(args.test_data, transformations, args.num_classes)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=2)
    # testing
    print('Ready to test network......')

    if args.huge_error:
        mkdir(os.path.join(args.save_dir, 'huge_error'))

    if args.collect_score:
        mkdir(os.path.join(args.save_dir, "collect_score"))

    if args.mode == 'reg':
        test_reg(model, test_loader, args)
    elif args.mode in ['cls', 'sord']:
        test_cls(model, test_loader, softmax, args)
