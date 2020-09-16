# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.init as init


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        # self.squeeze_bn = nn.BatchNorm2d(squeeze_planes)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)
        # self.cat_bn = nn.BatchNorm2d(expand1x1_planes + expand3x3_planes)

    def forward(self, x):
        x = self.squeeze(x)
        # x = self.squeeze_bn(x)
        x = self.squeeze_activation(x)
        x = torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)
        # x = self.cat_bn(x)
        return x


class SqueezeNet(nn.Module):

    def __init__(self, version='1_1', num_classes=1000):
        super(SqueezeNet, self).__init__()
        self.num_classes = num_classes
        if version == '1_0':
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == '1_1':
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                # nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 96, 512, 512),
                Fire(1024, 96, 512, 512)
            )
        else:
            # FIXME: Is this needed? SqueezeNet should only be called from the
            # FIXME: squeezenet1_x() functions
            # FIXME: This checking is not done for the other models
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1_0 or 1_1 expected".format(version=version))

        # Final convolution is initialized differently from the rest
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_x = nn.Linear(1024, self.num_classes)
        self.fc_y = nn.Linear(1024, self.num_classes)
        self.fc_z = nn.Linear(1024, self.num_classes)
        # final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        # self.fc_x = nn.Sequential(
        #     nn.Dropout(p=0.5),
        #     final_conv,
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveAvgPool2d((1, 1))
        # )
        # self.fc_y = nn.Sequential(
        #     nn.Dropout(p=0.5),
        #     final_conv,
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveAvgPool2d((1, 1))
        # )
        # self.fc_z = nn.Sequential(
        #     nn.Dropout(p=0.5),
        #     final_conv,
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveAvgPool2d((1, 1))
        # )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # if m is final_conv:
                #     init.normal_(m.weight, mean=0.0, std=0.01)
                # else:
                #     init.kaiming_uniform_(m.weight)
                # if m.bias is not None:
                #     init.constant_(m.bias, 0)

                init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, pahse='train'):
        x = self.features(x)
        x = self.global_pool(x).view(x.size(0), -1)
        v_x = self.fc_x(x)
        v_y = self.fc_y(x)
        v_z = self.fc_z(x)

        # v_x = self.fc_x(x)
        # v_x = v_x.view(x.size(0), -1)
        # v_y = self.fc_y(x)
        # v_y = v_y.view(x.size(0), -1)
        # v_z = self.fc_z(x)
        # v_z = v_z.view(x.size(0), -1)

        return v_x, v_y, v_z


def build_model():
    return SqueezeNet(version='1_1', num_classes=66)
