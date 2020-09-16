# -*- coding:utf-8 -*-
"""
训练resnet、mobilenet、shufflenet、squeezeNet的cls和sord：优化到分类网络
"""
import argparse
from tools import *
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.model_zoo as model_zoo
from torchvision import transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.models.resnet import model_urls as resnet_model_urls
from torchvision.models.mobilenet import model_urls as mobilenet_model_urls
from torchvision.models.shufflenetv2 import model_urls as shufflenetv2_model_urls
from torchvision.models.squeezenet import model_urls as squeezenet_model_urls


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
                        default=50, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
                        default=64, type=int)
    parser.add_argument('--lr', dest='lr', help='Base learning rate.',
                        default=0.00001, type=float)
    parser.add_argument('--lr_decay', dest='lr_decay', help='learning decay rate.',
                        default=1.0, type=float)
    parser.add_argument('--train_data', dest='train_data', help='Directory path for data.',
                        default='/home/pizza/dataset/300WLPQUAT', type=str)
    parser.add_argument('--valid_data', dest='valid_data', help='Directory path for data.',
                        default='/home/pizza/dataset/AFLW2000QUAT', type=str)
    parser.add_argument('--net', dest='net', help='net name for training', choices=['resnet', 'mobilenet', 'shufflenet', 'squeezenet'],
                        default='resnet', type=str)
    parser.add_argument('--loss', dest='loss', help='soft label loss function', choices=['BCE', 'KDJ'],
                        default='BCE', type=str)
    parser.add_argument('--save_dir', dest='save_dir', help='model saving path',
                        default='/home/pizza/results/HeadPoseNetExp', type=str)
    parser.add_argument('--unfreeze', dest='unfreeze', help='Choose after how many epochs to unfreeze all layers',
                        default=10, type=int)
    parser.add_argument('--mode', dest='mode', help='direct regression or cls or sord', choices=['reg', 'cls', 'sord'],
                        default='cls', type=str)
    parser.add_argument('--top_k', dest='top_k', help='top_k for cls2reg',
                        default=None, type=int)
    parser.add_argument('--num_classes', dest='num_classes', help='number of classify',
                        default=66, type=int)
    parser.add_argument('--alpha', dest='alpha', help='Regression loss coefficient.',
                        default=1., type=float)
    parser.add_argument('--cls2reg', dest='cls2reg', choices=['Expectation', 'Average'], help='number of classify',
                        default='Expectation', type=str)
    parser.add_argument('--width_mult', dest='width_mult', choices=[0.5, 1.0], help='mobilenet_v2 width_mult',
                        default=1.0, type=float)
    parser.add_argument('--input_size', dest='input_size', choices=[224, 192, 160, 128, 96], help='size of input images',
                        default=224, type=int)
    parser.add_argument('--squeeze_version', dest='squeeze_version', choices=['1_0', '1_1'], help='size of input images',
                        default='1_1', type=str)

    args = parser.parse_args()
    return args


def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.conv1, model.bn1]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param


def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param


def get_reg_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_angle]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param


def get_cls_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_x, model.fc_y, model.fc_z]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param


def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


def validation_reg(model, valid_loader):
    # validation
    with torch.no_grad():
        degrees_error = 0.
        count = 0.
        for j, (valid_img, valid_vector_label, _, _, _) in enumerate(valid_loader):
            valid_img = valid_img.cuda(0)

            valid_vector_label = valid_vector_label.cuda(0)
            pred_valid_vector = model(valid_img)
            pred_valid_vector = norm_vector(pred_valid_vector).cuda(0)

            cos_value = vector_cos(pred_valid_vector, valid_vector_label)
            degrees_error += torch.mean(torch.acos(cos_value) * 180 / np.pi)
            count += 1.
    return degrees_error / count


def validation_cls(model, valid_loader, softmax, args):
    # validation
    with torch.no_grad():
        degrees_error = 0.
        count = 0.
        for j, (valid_img, cls_label, vector_label, _, _, _) in enumerate(valid_loader):
            valid_img = valid_img.cuda(0)
            vector_label = vector_label.cuda(0)

            # get x,y,z cls predictions
            x_cls_pred, y_cls_pred, z_cls_pred = model(valid_img)

            # get prediction vector(get continue value from classify result)
            _, _, _, vector_pred = classify2vector(x_cls_pred, y_cls_pred, z_cls_pred, softmax,
                                                   num_classes=args.num_classes,
                                                   top_k=args.top_k, cls2reg=args.cls2reg)

            # get validation degrees error
            cos_value = utils.vector_cos(vector_pred, vector_label)
            degrees_error += torch.mean(torch.acos(cos_value) * 180 / np.pi)
            count += 1.
    return degrees_error / count


def train_reg(model, train_loader, valid_loader, reg_criterion, output_string, args):
    lr = args.lr
    steps = 0
    for epoch in range(num_epochs):
        if args.unfreeze == 0:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            if epoch >= args.unfreeze:
                optimizer = torch.optim.Adam([{'params': get_ignored_params(model), 'lr': lr},
                                              {'params': get_non_ignored_params(model), 'lr': lr},
                                              {'params': get_reg_fc_params(model), 'lr': lr}],
                                             lr=args.lr)
            else:
                optimizer = torch.optim.Adam([{'params': get_ignored_params(model), 'lr': 0},
                                              {'params': get_non_ignored_params(model), 'lr': lr},
                                              {'params': get_reg_fc_params(model), 'lr': lr * 5}],
                                             lr=args.lr)
        lr *= args.lr_decay
        best_degrees_error = 180
        for i, (images, label_vector, name) in enumerate(train_loader):
            steps += 1
            images = images.cuda(0)

            label_vector = label_vector.cuda(0)
            pred_vector = model(images)
            pred_vector = utils.norm_vector(pred_vector).cuda(0)

            loss = reg_criterion(pred_vector * 10, label_vector * 10)
            cos_value = utils.vector_cos(pred_vector, label_vector)
            degrees_error = torch.mean(torch.acos(cos_value) * 180 / np.pi)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                msg = 'Epoch [%d/%d], Iter [%d/%d] mse_loss: %.4f degrees_loss:%.3f' % (
                    epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.item(), degrees_error.item())
                logger.logger.info(msg)
                valid_degrees_error = validation_reg(model, valid_loader)

                # writer summary
                writer.add_scalar("train loss", loss, steps)
                writer.add_scalar("valid loss", valid_degrees_error, steps)

                if valid_degrees_error < best_degrees_error:
                    logger.logger.info("A better validation degrees error {}".format(valid_degrees_error))
                    best_degrees_error = valid_degrees_error
                    torch.save(model.state_dict(), os.path.join(snapshots_dir, output_string + '_epoch_' + str(epoch + 1) + '.pkl'))


def train_cls(model, train_loader, valid_loader, cls_criterion, reg_criterion, output_string, args):
    lr = args.lr
    steps = 0
    for epoch in range(num_epochs):
        if args.unfreeze == 0:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            if epoch >= args.unfreeze:
                optimizer = torch.optim.Adam([{'params': get_ignored_params(model), 'lr': lr},
                                              {'params': get_non_ignored_params(model), 'lr': lr},
                                              {'params': get_cls_fc_params(model), 'lr': lr}],
                                             lr=args.lr)
            else:
                optimizer = torch.optim.Adam([{'params': get_ignored_params(model), 'lr': 0},
                                              {'params': get_non_ignored_params(model), 'lr': lr},
                                              {'params': get_cls_fc_params(model), 'lr': lr * 5}],
                                             lr=args.lr)
        lr *= args.lr_decay
        best_degrees_error = 180
        for i, (images, classify_label, vector_label, name) in enumerate(train_loader):
            steps += 1
            images = images.cuda(0)
            classify_label = classify_label.cuda(0)
            vector_label = vector_label.cuda(0)

            # get x,y,z cls label
            x_cls_label = classify_label[:, 0]
            y_cls_label = classify_label[:, 1]
            z_cls_label = classify_label[:, 2]

            # get x,y,z continue label
            x_reg_label = vector_label[:, 0]
            y_reg_label = vector_label[:, 1]
            z_reg_label = vector_label[:, 2]

            # get x,y,z cls predictions
            x_cls_pred, y_cls_pred, z_cls_pred = model(images)

            # CrossEntry loss(for classify)
            x_cls_loss = cls_criterion(x_cls_pred, x_cls_label)
            y_cls_loss = cls_criterion(y_cls_pred, y_cls_label)
            z_cls_loss = cls_criterion(z_cls_pred, z_cls_label)

            # get prediction vector(get continue value from classify result)
            x_reg_pred, y_reg_pred, z_reg_pred, vector_pred = classify2vector(x_cls_pred, y_cls_pred, z_cls_pred, softmax,
                                                                              num_classes=args.num_classes,
                                                                              top_k=args.top_k,
                                                                              cls2reg=args.cls2reg)
            # Regression loss
            x_reg_loss = reg_criterion(x_reg_pred, x_reg_label)
            y_reg_loss = reg_criterion(y_reg_pred, y_reg_label)
            z_reg_loss = reg_criterion(z_reg_pred, z_reg_label)

            # Total loss
            x_loss = x_cls_loss + args.alpha * x_reg_loss
            y_loss = y_cls_loss + args.alpha * y_reg_loss
            z_loss = z_cls_loss + args.alpha * z_reg_loss

            # backward
            loss_seq = [x_loss, y_loss, z_loss]
            grad_seq = [torch.ones(1).cuda(0) for _ in range(len(loss_seq))]
            optimizer.zero_grad()
            torch.autograd.backward(loss_seq, grad_seq)
            optimizer.step()

            # get degrees error
            cos_value = utils.vector_cos(vector_pred, vector_label)
            degrees_error = torch.mean(torch.acos(cos_value) * 180 / np.pi)

            if (i + 1) % 100 == 0:
                msg = "Epoch: %d/%d | Iter: %d/%d | x_loss: %.6f | y_loss: %.6f | z_loss: %.6f | degree_error:%.3f" % (
                    epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, x_loss.item(), y_loss.item(),
                    z_loss.item(), degrees_error.item())
                logger.logger.info(msg)
                valid_degrees_error = validation_cls(model, valid_loader, softmax, args)

                # writer summary
                writer.add_scalar("x classify loss", x_cls_loss, steps)
                writer.add_scalar("y classify loss", y_cls_loss, steps)
                writer.add_scalar("z classify loss", z_cls_loss, steps)
                writer.add_scalar("classify average loss", (x_cls_loss + y_cls_loss + z_cls_loss) / 3, steps)
                writer.add_scalar("x total loss", x_loss, steps)
                writer.add_scalar("y total loss", x_loss, steps)
                writer.add_scalar("z total loss", x_loss, steps)
                writer.add_scalar("total average loss", (x_loss + y_loss + z_loss) / 3, steps)
                writer.add_scalar("train degrees error", degrees_error, steps)
                writer.add_scalar("valid degrees error", valid_degrees_error, steps)

                if valid_degrees_error < best_degrees_error:
                    logger.logger.info("A better validation degrees error {}".format(valid_degrees_error))
                    best_degrees_error = valid_degrees_error
                    torch.save(model.state_dict(), os.path.join(snapshots_dir, output_string + '_epoch_' + str(epoch + 1) + '.pkl'))


if __name__ == "__main__":
    args = parse_args()
    if args.mode == 'reg':
        output_string = args.net + '_reg_width_mult%s_%s%s' % (str(args.width_mult).replace('.', '_'),
                                                               str(args.input_size))
    else:
        if args.top_k is None:
            output_string = args.net + "_%s_%s_top%s_alpha%s_width_mult%s_%s_%s" % (args.mode,
                                                                                    args.num_classes,
                                                                                    args.top_k,
                                                                                    str(args.alpha).replace('.', '_'),
                                                                                    str(args.width_mult).replace('.', '_'),
                                                                                    str(args.input_size),
                                                                                    args.loss)
        else:
            output_string = args.net + "_%s_%s_top%s_%s_alpha%s_width_mult%s_%s_%s" % (args.mode,
                                                                                       args.num_classes, args.top_k, args.cls2reg,
                                                                                       str(args.alpha).replace('.', '_'),
                                                                                       str(args.width_mult).replace('.', '_'),
                                                                                       str(args.input_size),
                                                                                       args.loss)

    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr

    # mkdir
    project_dir = os.path.join(args.save_dir, output_string)
    utils.mkdir(project_dir)
    snapshots_dir = os.path.join(project_dir, 'snapshots')
    utils.mkdir(snapshots_dir)
    summary_dir = os.path.join(project_dir, 'summary')
    utils.mkdir(summary_dir)
    log_path = os.path.join(project_dir, output_string + '.log')

    # create writer summary and log
    writer = SummaryWriter(logdir=summary_dir)
    logger = Logger(log_path, 'info')

    # print parameters
    logger.logger.info("Parameters".center(100, '='))
    logger.logger.info(
        "\nnum_epochs: %s\nbatch_size: %s\nunfreeze: %s\nmode: %s\nnet: %s\ntop_k: %s\nnum_classes: %s\nwidth_mult: %s\nsqueeze_version: %s\ninput_size: %s\nalpha: %s\ncls2reg: %s\nlearning_rate: %s\nlr_decay: %s" % (
            args.num_epochs, args.batch_size, args.unfreeze, args.mode, args.net, args.top_k, args.num_classes, args.width_mult, args.squeeze_version,
            args.input_size,
            args.alpha, args.cls2reg, args.lr, args.lr_decay))

    # cls and sord
    if args.mode in ['cls', 'sord']:
        if args.net == 'resnet':
            model = ResNetCls(layers=[3, 4, 6, 3], num_classes=args.num_classes)
            load_filtered_state_dict(model, model_zoo.load_url(resnet_model_urls['resnet50']))
        elif args.net == 'mobilenet':
            model = MobileNetV2Cls(num_classes=args.num_classes, width_mult=args.width_mult)
            if args.width_mult == 1.0:
                logger.logger.info("Loading pretrained weight".center(100, '='))
                load_filtered_state_dict(model, model_zoo.load_url(mobilenet_model_urls['mobilenet_v2']))
        elif args.net == 'shufflenet':
            model = ShuffleNetV2Cls(scale=args.width_mult, num_classes=args.num_classes)
            if args.width_mult == 0.5:
                load_filtered_state_dict(model, model_zoo.load_url(shufflenetv2_model_urls['shufflenetv2_x0.5']))
            elif args.width_mult == 1.0:
                load_filtered_state_dict(model, model_zoo.load_url(shufflenetv2_model_urls['shufflenetv2_x1.0']))
        elif args.net == 'squeezenet':
            model = SqueezeNetCls(version=args.squeeze_version, num_classes=args.num_classes)
            if args.squeeze_version == '1_0':
                load_filtered_state_dict(model, model_zoo.load_url(squeezenet_model_urls['squeezenet1_0']))
            elif args.squeeze_version == '1_1':
                load_filtered_state_dict(model, model_zoo.load_url(squeezenet_model_urls['squeezenet1_1']))
    # reg
    elif args.mode == 'reg':
        if args.net == 'resnet':
            model = ResNetReg(layers=[3, 4, 6, 3], num_classes=3)
            load_filtered_state_dict(model, model_zoo.load_url(resnet_model_urls['resnet50']))
        elif args.net == 'mobilenet':
            model = MobileNetV2Reg(num_classes=3, width_mult=args.width_mult)
            if args.width_mult == 1.0:
                logger.logger.info("Loading pretrained weight")
                load_filtered_state_dict(model, model_zoo.load_url(mobilenet_model_urls['mobilenet_v2']))
        elif args.net == 'shufflenet':
            model = ShuffleNetV2Reg(scale=args.width_mult, num_classes=args.num_classes)
            if args.width_mult == 0.5:
                load_filtered_state_dict(model, model_zoo.load_url(shufflenetv2_model_urls['shufflenetv2_x0.5']))
            elif args.width_mult == 1.0:
                load_filtered_state_dict(model, model_zoo.load_url(shufflenetv2_model_urls['shufflenetv2_x1.0']))
        elif args.net == 'squeezenet':
            model = SqueezeNetReg(version=args.squeeze_version, num_classes=args.num_classes)
            if args.squeeze_version == '1_0':
                load_filtered_state_dict(model, model_zoo.load_url(squeezenet_model_urls['squeezenet1_0']))
            elif args.squeeze_version == '1_1':
                load_filtered_state_dict(model, model_zoo.load_url(squeezenet_model_urls['squeezenet1_1']))

    # loading data
    logger.logger.info('Loading data'.center(100, '='))

    # define transformations
    train_transformations = transforms.Compose([transforms.Resize(int(np.ceil(args.input_size * 1.0714))),
                                                transforms.RandomCrop(args.input_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    valid_transformations = transforms.Compose([transforms.Resize(args.input_size),
                                                transforms.RandomCrop(args.input_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # load train dataset
    if args.mode == 'reg':
        train_dataset = TrainDataSetReg(args.train_data, train_transformations)
        valid_dataset = TestDataSetReg(args.valid_data, valid_transformations, length=args.batch_size * 5)
    elif args.mode == 'cls':
        train_dataset = TrainDataSetCls(args.train_data, train_transformations, args.num_classes)
        valid_dataset = TestDataSetCls(args.valid_data, valid_transformations, args.num_classes, length=args.batch_size * 5)
    elif args.mode == 'sord':
        train_dataset = TrainDataSetSORD(args.train_data, train_transformations, args.num_classes)
        valid_dataset = TestDataSetSORD(args.valid_data, valid_transformations, args.num_classes, length=args.batch_size * 5)

    # initialize data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # load validation dataset
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # loss function
    if args.mode == 'cls':
        cls_criterion = nn.CrossEntropyLoss().cuda(0)
    elif args.mode == 'sord':
        if args.loss == 'BCE':
            cls_criterion = nn.BCEWithLogitsLoss().cuda(0)
        elif args.loss == 'KDJ':
            cls_criterion = SoftLabelLoss().cuda(0)
        else:
            print("soft label loss function should be BCE or KDJ")

    reg_criterion = nn.MSELoss().cuda(0)
    softmax = nn.Softmax(dim=1).cuda(0)
    model.cuda(0)

    # training
    logger.logger.info('Ready to train network'.center(100, '='))
    if args.mode == 'reg':
        train_reg(model, train_loader, valid_loader, reg_criterion, output_string, args)
    elif args.mode in ['cls', 'sord']:
        train_cls(model, train_loader, valid_loader, cls_criterion, reg_criterion, output_string, args)
