# -*- coding:utf-8 -*-
"""
    rotation functions script
"""
import numpy as np
from math import cos
from math import sin


class Rotation(object):
    def __init__(self):
        pass

    @staticmethod
    def euler2dcm(pitch, yaw, roll):
        """
        calculate right hand dcm with euler angle by order xyz
        :param pitch: float
        :param yaw: float
        :param roll: float
        :return: np.mat(3,3)
        """
        rx = np.mat([[1, 0, 0],
                     [0, cos(pitch), sin(pitch)],
                     [0, -sin(pitch), cos(pitch)]])
        ry = np.mat([[cos(yaw), 0, -sin(yaw)],
                     [0, 1, 0],
                     [sin(yaw), 0, cos(yaw)]])
        rz = np.mat([[cos(roll), sin(roll), 0],
                     [-sin(roll), cos(roll), 0],
                     [0, 0, 1]])
        dcm = rx * ry * rz
        return dcm

    @staticmethod
    def transform_dcm(dcm, axis='x'):
        """
        transfer right hand dcm to left hand dcm
        :param dcm: np.mat(3,3)
        :param axis: str
        :return: np.mat(3,3)
        """
        if axis == 'x':
            s = np.mat([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        elif axis == 'y':
            s = np.mat([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        else:
            s = np.mat([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
        return s * dcm * s

    @staticmethod
    def dcm2quat(dcm):
        """
        transfer dcm to quat
        :param dcm: np.mat(3,3)
        :return: np.ndarray(,3)
        """
        w2 = (dcm[0, 0] + dcm[1, 1] + dcm[2, 2] + 1) / 4
        if w2 > 0:
            x = (dcm[2, 1] - dcm[1, 2]) / (4 * np.sqrt(w2))
            y = (dcm[0, 2] - dcm[2, 0]) / (4 * np.sqrt(w2))
            z = (dcm[1, 0] - dcm[0, 1]) / (4 * np.sqrt(w2))
        elif w2 == 0:
            x2 = -(dcm[1, 1] + dcm[2, 2]) / 2
            if x2 > 0:
                y2 = dcm[0, 1] / (2 * np.sqrt(x2))
                z2 = dcm[0, 2] / (2 * np.sqrt(x2))
            else:
                y2 = (1 - dcm[2, 2]) / 2
                if y2 > 0:
                    z2 = dcm[1, 2] / (2 * np.sqrt(y2))
                else:
                    z2 = 1
            x = np.sqrt(x2)
            y = np.sqrt(y2)
            z = np.sqrt(z2)
        w = np.sqrt(w2)

        quat = np.array([x, y, z, w]).astype(np.float32)
        return quat

    @staticmethod
    def quat2dcm(quat):
        """
        transfer quat to left hand dcm
        :param quat:np.ndarray(,3)[x,y,z,w]
        :return:
        """
        e1, e2, e3, e0 = quat
        dcm = np.mat([[e0 ** 2 + e1 ** 2 - e2 ** 2 - e3 ** 2, 2 * (e1 * e2 - e0 * e3), 2 * (e1 * e3 + e0 * e2)],
                      [2 * (e1 * e2 + e0 * e3), e0 ** 2 - e1 ** 2 + e2 ** 2 - e3 ** 2, 2 * (e2 * e3 - e0 * e1)],
                      [2 * (e1 * e3 - e0 * e2), 2 * (e2 * e3 + e0 * e1), e0 ** 2 - e1 ** 2 - e2 ** 2 + e3 ** 2]])
        return dcm
