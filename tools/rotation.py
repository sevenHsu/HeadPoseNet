import numpy as np
from math import cos,sin
import scipy.io as sio
import cv2
import os


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

def orientation(pitch,yaw,roll):
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

def inverse_orientation(pitch,yaw,roll):
    rx = np.mat([[1, 0, 0],
                [0, cos(pitch), sin(pitch)],
                [0, -sin(pitch), cos(pitch)]])
    ry = np.mat([[cos(yaw), 0, -sin(yaw)],
                [0, 1, 0],
                [sin(yaw), 0, cos(yaw)]])
    rz = np.mat([[cos(roll), sin(roll), 0],
                [-sin(roll), cos(roll), 0],
                [0, 0, 1]])
    dcm = rz * ry * rx
    return dcm

def rotation(pitch,yaw,roll):
    rx = np.mat([[1, 0, 0],
                [0, cos(pitch), -sin(pitch)],
                [0, sin(pitch), cos(pitch)]])
    ry = np.mat([[cos(yaw), 0, sin(yaw)],
                [0, 1, 0],
                [-sin(yaw), 0, cos(yaw)]])
    rz = np.mat([[cos(roll), -sin(roll), 0],
                [sin(roll), cos(roll), 0],
                [0, 0, 1]])
    dcm = rz*ry*rx
    return dcm


if __name__ == "__main__":
    image_set= set([i[:-4] for i in os.listdir("../300wlp_mini")])
    for img_name in image_set:
        def get_pose(path):
            mat = sio.loadmat(path)
            euler_angles = mat["Pose_Para"][0][:5]
            return euler_angles
        def draw_axis(p,y,r,tx,ty,img):
            h,w,_=img.shape
            s=np.mat([[1,0,0],[0,-1,0],[0,0,1]])

            # dcm_o = s*orientation(p,y,r)*s
            # vx=dcm_o*np.mat([[1],[0],[0]])*50
            # vy=dcm_o*np.mat([[0],[1],[0]])*50
            # vz=dcm_o*np.mat([[0],[0],[1]])*50

            # dcm_i = inverse_orientation(p,y,r)
            # vx=np.array(np.mat([1,0,0])*dcm_i)*50
            # vy=np.array(np.mat([0,1,0])*dcm_i)*50
            # vz=np.array(np.mat([0,0,1])*dcm_i)*50

            dcm_r = s*rotation(p,y,r)*s
            vx=np.array(np.mat([1,0,0])*dcm_r)*50
            vy=np.array(np.mat([0,1,0])*dcm_r)*50
            vz=np.array(np.mat([0,0,1])*dcm_r)*50
            
            cv2.line(img, (int(tx), int(ty)), (int(vx[0][0]+tx), int(vx[0][1]+ty)), (0, 0, 255), 2)
            cv2.line(img, (int(tx), int(ty)), (int(vy[0][0]+tx), int(vy[0][1]+ty)), (0, 255, 0), 2)
            cv2.line(img, (int(tx), int(ty)), (int(vz[0][0]+tx), int(vz[0][1]+ty)), (255, 0, 0), 2)
            str_euler = "p:"+str(int(p*180/np.pi))+"y:"+str(int(y*180/np.pi))+"r:"+str(int(r*180/np.pi))
            cv2.putText(img, str_euler, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        img = cv2.imread("../300wlp_mini/"+img_name+".jpg")
        mat_path = "../300wlp_mini/"+img_name+".mat"
        p,y,r,tx,ty = get_pose(mat_path)
        draw_axis(p,y,r,tx,ty,img)

        cv2.imwrite("../300wlp_mini_show/"+img_name+".jpg",img)
