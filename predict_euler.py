import os
import tqdm
import torch
import argparse
import torchvision
import numpy as np
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data import DataLoader
from tools import utils, nets, dataset_euler


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--data_dir', dest='data_dir', help='data dir path for predicting images',
                        default='/train/trainset/1/dms_filter/dms', type=str)
    parser.add_argument('--filename_list', dest='filename_list', help='filename_list contain img_name and bbox',
                        default='', type=str)
    parser.add_argument('--snapshot', dest='snapshot', help='Name of model snapshot.',
                        default='/train/trainset/1/results/euler/snapshots/300aug_epoch_25.pkl', type=str)
    parser.add_argument('--out_path', dest='out_path', help='file path for saving predict results',
                        default='/train/trainset/1/dms_filter/dms_pose.txt', type=str)
    parser.add_argument('--batch_size', dest='batch_size', help='batch size for predict',
                        default=64, type=int)

    args = parser.parse_args()

    return args


def predict(model, data_loader, out_path):
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(0)

    fw = open(out_path, 'w')
    for j, (images, image_names) in enumerate(tqdm.tqdm(data_loader)):
        images = images.cuda(0)

        yaw, pitch, roll = model(images)

        # Continuous predictions
        yaw_predicted = utils.softmax_temperature(yaw.data, 1)
        pitch_predicted = utils.softmax_temperature(pitch.data, 1)
        roll_predicted = utils.softmax_temperature(roll.data, 1)

        yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1) * 3 - 99
        roll_predicted = torch.sum(roll_predicted * idx_tensor, 1) * 3 - 99

        # Save first image in batch with pose cube or axis.
        for i in range(len(image_names)):
            image_i = image_names[i]
            yaw_i = yaw_predicted[i].item() * np.pi / 180
            pitch_i = pitch_predicted[i].item() * np.pi / 180
            roll_i = roll_predicted[i].item() * np.pi / 180

            fw.write('%s %.5f %.5f %.5f\n' % (image_i, pitch_i, yaw_i, roll_i))
    fw.close()


if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    gpu = args.gpu_id
    filename_list = args.filename_list

    # loading model
    print('Loading model')
    snapshot_path = args.snapshot
    model = nets.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)
    model.cuda(0)
    model.eval()

    # loading data
    print("loading data")
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset = dataset_euler.PredictData(args.data_dir, args.filename_list, transform)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=args.batch_size,
                                              num_workers=8)
    predict(model, data_loader, args.out_path)
