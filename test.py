# -*- coding:utf-8 -*-
import os
import tqdm
import utils
import torch
import argparse
import cv2 as cv
import numpy as np
import torch.nn as nn
from net import MobileNetV2
from dataset import loadData


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--test_data', dest='test_data', help='Directory path for data.',
                        default='/home/pizza/dataset/AFLW2000QUAT', type=str)
    parser.add_argument('--snapshot', dest='snapshot', help='Name of model snapshot.',
                        default='', type=str)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
                        default=64, type=int)
    parser.add_argument('--degree_error_limit', dest='degree_error_limit', help='degrees error for calc cs',
                        default=10, type=int)
    parser.add_argument('--save_dir', dest='save_dir', help='directory for saving drawn pic',
                        default='/home/pizza/results/MobileNetV2_1.0_classes_66_input_224', type=str)
    parser.add_argument('--show_front', dest='show_front', help='show front or not',
                        default=True, type=bool)
    parser.add_argument('--analysis', dest='analysis', help='analysis result or not',
                        default=True, type=bool)
    parser.add_argument('--collect_score', dest='collect_score', help='show huge error or not',
                        default=True, type=bool)
    parser.add_argument('--num_classes', dest='num_classes', help='number of classify',
                        default=66, type=int)
    parser.add_argument('--width_mult', dest='width_mult', choices=[0.5, 1.0], help='mobilenet_v2 width_mult',
                        default=1.0, type=float)
    parser.add_argument('--input_size', dest='input_size', choices=[224, 192, 160, 128, 96], help='size of input images',
                        default=224, type=int)

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
    # if 'DMS_TEST_DATA' in args.test_data.split('/'):
    #     gtx, gty, gtz = vector_label
    #     utils.draw_front(img, gtx, gty, tdx=start_x, tdy=start_y, size=100, color=(0, 255, 0))

    # draw GT axis
    # elif 'AFLW2000QUAT' in args.test_data.split('/'):
    #     pitch, yaw, roll = angle_label
    #     utils.draw_axis(img, pitch, yaw, roll, tdx=start_x, tdy=start_y, size=100)

    # draw face bbox
    # utils.draw_bbox(img, pt2d)

    # draw pred attention vector with red
    utils.draw_front(img, predx, predy, tdx=start_x, tdy=start_y, size=100, color=(0, 0, 255))

    cv.imwrite(os.path.join(save_dir, img_name), img)


def test(model, test_loader, softmax, args):
    if args.analysis:
        utils.mkdir(os.path.join(args.save_dir, 'analysis'))
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
            _, _, _, pred_vector = utils.classify2vector(x_cls_pred, y_cls_pred, z_cls_pred, softmax, args.num_classes)

            # Mean absolute error
            cos_value = utils.vector_cos(pred_vector, vector_label)
            degrees_error = torch.acos(cos_value) * 180 / np.pi

            # save euler angle and degrees error to loss_dict
            if args.analysis:
                for k in range(len(angle_label)):
                    loss_dict['img_name'].append(names[k])
                    loss_dict['angles'].append(angle_label[k].tolist())  # pitch,yaw,roll
                    loss_dict['degree_error'].append(float(degrees_error[k]))

            # collect error
            error += torch.sum(degrees_error)
            score += torch.sum(utils.degress_score(cos_value, args.degree_error_limit))

            total += vector_label.size(0)

            # Save first image in batch with pose cube or axis.
            if args.show_front:
                utils.mkdir(os.path.join(args.save_dir, 'show_front'))
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
        utils.show_loss_distribute(loss_dict, os.path.join(args.save_dir, 'analysis'), os.path.basename(args.snapshot).split('.')[0])

    # save collect score curve
    if args.collect_score:
        print("analysis collect score")
        utils.collect_score(loss_dict, os.path.join(args.save_dir, "collect_score"))


if __name__ == '__main__':
    args = parse_args()

    utils.mkdir(args.save_dir)

    # cls and sord
    print("Loading model weight......")
    model = MobileNetV2(num_classes=args.num_classes)

    saved_state_dict = torch.load(args.snapshot)
    model.load_state_dict(saved_state_dict)
    model.cuda(0)

    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    softmax = nn.Softmax(dim=1).cuda(0)

    # test dataLoader
    test_loader = loadData(args.test_data, args.input_size, args.batch_size, args.num_classes, False)

    # testing
    print('Ready to test network......')

    if args.collect_score:
        utils.mkdir(os.path.join(args.save_dir, "collect_score"))
    test(model, test_loader, softmax, args)
