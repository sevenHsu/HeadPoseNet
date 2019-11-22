# -*- coding:utf-8 -*-
"""
    training HeadPoseNet
"""
import os
import utils
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from log import Logger
from dataset import loadData
from net import MobileNetV2
from tensorboardX import SummaryWriter
from torchvision.models.mobilenet import model_urls


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description="Head Pose Estimation using HeadPoseNet")
    parser.add_argument("--epochs", dest="epochs", help="Maximum number of training epochs.",
                        default=50, type=int)
    parser.add_argument("--batch_size", dest="batch_size", help="batch size",
                        default=16, type=int)
    parser.add_argument("--lr", dest="lr", help="Base learning rate",
                        default=0.00001, type=float)
    parser.add_argument("--lr_decay", dest="lr_decay", help="learning rate decay rate",
                        default=1.0, type=float)
    parser.add_argument("--save_dir", dest="save_dir", help="directory path of saving results",
                        default='/home/pizza/results', type=str)
    parser.add_argument("--train_data", dest="train_data", help="directory path of train dataset",
                        default="/home/pizza/dataset/300WLPQUAT", type=str)
    parser.add_argument("--valid_data", dest="valid_data", help="directory path of valid dataset",
                        default="/home/pizza/dataset/AFLW2000QUAT", type=str)
    parser.add_argument("--snapshot", dest="snapshot", help="pre trained weight path",
                        default="", type=str)
    parser.add_argument("--unfreeze", dest="unfreeze", help="unfreeze some layer after several epochs",
                        default=10, type=int)
    parser.add_argument("--num_classes", dest="num_classes", help="number of classify",
                        default=66, type=int)
    parser.add_argument("--alpha", dest="alpha", help="ragression loss coefficient",
                        default=1., type=float)
    parser.add_argument("--width_mult", dest="width_mult", choices=[0.5, 1.0], help="mobile V2 width_mult",
                        default=1., type=float)
    parser.add_argument("--input_size", dest="input_size", choices=[224, 192, 160, 128, 96], help="size of input images",
                        default=224, type=int)
    args = parser.parse_args()
    return args


def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.features]
    for i in range(len(b) - 2):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param


def get_cls_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.features[14:], model.fc_x, model.fc_y, model.fc_z]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param


def valid(model, valid_loader, softmax):
    with torch.no_grad():
        degrees_error = 0.
        count = 0.
        for j, (valid_img, cls_label, vector_label, _,) in enumerate(valid_loader):
            valid_img = valid_img.cuda(0)
            vector_label = vector_label.cuda(0)

            # get x,y,z cls predictions
            x_cls_pred, y_cls_pred, z_cls_pred = model(valid_img)

            # get prediction vector(get continue value from classify result)
            _, _, _, vector_pred = utils.classify2vector(x_cls_pred, y_cls_pred, z_cls_pred, softmax, args.num_classes, )

            # get validation degrees error
            cos_value = utils.vector_cos(vector_pred, vector_label)
            degrees_error += torch.mean(torch.acos(cos_value) * 180 / np.pi)
            count += 1.
    return degrees_error / count


def train():
    """
    :return:
    """
    # create model
    model = MobileNetV2(args.num_classes, width_mult=args.width_mult)

    # loading pre trained weight
    logger.logger.info("Loading PreTrained Weight".center(100, '='))
    utils.load_filtered_stat_dict(model, model_zoo.load_url(model_urls["mobilenet_v2"]))

    # loading data
    logger.logger.info("Loading data".center(100, '='))
    train_data_loader, valid_data_loader = loadData(args.train_data, args.input_size, args.batch_size, args.num_classes)

    # initialize loss function
    cls_criterion = nn.BCEWithLogitsLoss().cuda(0)
    reg_criterion = nn.MSELoss().cuda(0)
    softmax = nn.Softmax(dim=1).cuda(0)
    model.cuda(0)

    # training
    logger.logger.info("Training".center(100, '='))

    # initialize learning rate and step
    lr = args.lr
    step = 0
    for epoch in range(args.epochs + 1):
        if epoch > args.unfreeze:
            optimizer = torch.optim.Adam([{"params": get_non_ignored_params(model), "lr": lr},
                                          {"params": get_cls_fc_params(model), "lr": lr}], lr=args.lr)
        else:
            optimizer = torch.optim.Adam([{"params": get_non_ignored_params(model), "lr": 0},
                                          {"params": get_cls_fc_params(model), "lr": lr * 5}], lr=args.lr)
        lr = lr * args.lr_decay
        min_degree_error = 180.
        for i, (images, classify_label, vector_label, name) in enumerate(train_data_loader):
            step += 1
            images = images.cuda(0)
            classify_label = classify_label.cuda(0)
            vector_label = vector_label.cuda(0)

            # inference
            x_cls_pred, y_cls_pred, z_cls_pred = model(images)
            logits = [x_cls_pred, y_cls_pred, z_cls_pred]
            loss, degree_error = utils.computeLoss(classify_label, vector_label, logits, softmax, cls_criterion, reg_criterion, args)

            # backward
            grad = [torch.ones(1).cuda(0) for _ in range(3)]
            optimizer.zero_grad()
            torch.autograd.backward(loss, grad)
            optimizer.step()

            # save training log and weight
            if (i + 1) % 100 == 0:
                msg = "Epoch: %d/%d | Iter: %d/%d | x_loss: %.6f | y_loss: %.6f | z_loss: %.6f | degree_error:%.3f" % (
                    epoch, args.epochs, i + 1, len(train_data_loader.dataset) // args.batch_size, loss[0].item(), loss[1].item(),
                    loss[2].item(), degree_error.item())
                logger.logger.info(msg)
                valid_degree_error = valid(model, valid_data_loader, softmax)

                # writer summary
                writer.add_scalar("train degrees error", degree_error, step)
                writer.add_scalar("valid degrees error", valid_degree_error, step)

                # saving snapshot
                if valid_degree_error < min_degree_error:
                    min_degree_error = valid_degree_error
                    logger.logger.info("A better validation degrees error {}".format(valid_degree_error))
                    torch.save(model.state_dict(), os.path.join(snapshot_dir, output_string + '_epoch_' + str(epoch) + '.pkl'))


if __name__ == "__main__":
    args = parse_args()
    output_string = "MobileNetV2_%s_classes_%s_input_%s" % (args.width_mult, args.num_classes, args.input_size)

    # mkdir
    project_dir = os.path.join(args.save_dir, output_string)
    utils.mkdir(project_dir)
    snapshot_dir = os.path.join(project_dir, "snapshot")
    utils.mkdir(snapshot_dir)
    summary_dir = os.path.join(project_dir, "summary")
    utils.mkdir(summary_dir)
    log_path = os.path.join(project_dir, "training.log")

    # create summary writer and log
    writer = SummaryWriter(log_dir=summary_dir)
    logger = Logger(log_path, 'info')

    # print parameters
    logger.logger.info("Parameters".center(100, '='))
    logger.logger.info("\ninput_size:%d\nunfreeze:%d\nnum_classes:%d\nepochs:%d\nbatch_size:%d\nlr:%f\nlr_decay:%f\nalpha:%f\n" % (
        args.input_size, args.unfreeze, args.num_classes, args.epochs, args.batch_size, args.lr, args.lr_decay, args.alpha))

    # run train function
    train()
