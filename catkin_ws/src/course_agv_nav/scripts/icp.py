#!/usr/bin/env python

from scipy.spatial import distance
import rospy
import tf
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import numpy as np
import math
from sklearn.neighbors import NearestNeighbors
import cv2
import matplotlib.pyplot as plt

class NeighBor(object):
    def __init__(self):
        self.distances = []
        self.src_indices = []
        self.tar_indices = []


class ICP(object):
    def __init__(self):
        # max iterations
        self.max_iter = rospy.get_param('/icp/max_iter', 20)
        # distance threshold for filter the matching points
        self.dis_th = rospy.get_param('/icp/dis_th', 0.1)
        # tolerance to stop icp
        self.tolerance = rospy.get_param('/icp/tolerance',0)
        # min match
        self.min_match = rospy.get_param('/icp/min_match',2)
        # self.count = 0
        print("ICP params:")
        print("dis_th:", self.dis_th)
        print("max_iter:", self.max_iter)
        
    def process(self, tar_pc, src_pc):
        # self.count += 1
        # if self.count % 200 == 0:
        #     plt.axis("equal")
        #     plt.plot(src_pc[0], src_pc[1],marker='*',linestyle='')
        #     plt.show()
        T_acc = np.identity(3)
        A = np.array(src_pc)[:2].T
        B = np.array(tar_pc)[:2].T
        m = A.shape[1]
        src = np.ones((m+1, A.shape[0]))
        dst = np.ones((m+1, B.shape[0]))
        src[:m, :] = np.copy(A.T)
        dst[:m, :] = np.copy(B.T)
        mean_error = 0
        prev_error = 0
        delta_error = 0

        for i in range(self.max_iter):
            distances, src_indices, tar_indices = self.findNearest(src[:m, :].T, dst[:m, :].T)
            mean_error = np.mean(distances)
            std_error = np.std(distances)
            T = self.getTransform(src[:m, src_indices].T, dst[:m, tar_indices].T)
            T_acc = np.dot(T, T_acc)
            src = np.dot(T, src)
            delta_error = mean_error - prev_error

            if abs(delta_error) < self.tolerance:
                break
            prev_error = mean_error

        return T_acc, len(distances)

    def findNearest(self, src, tar):
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(tar)
        try:
            distances, indices = neigh.kneighbors(src, return_distance=True)
        except:
            print('neighbors failed')
        src_indices = []
        tar_indices = []
        filtered_dis = []
        distances = distances.ravel().tolist()
        indices = indices.ravel()
        for i in range(len(distances)):
            if (distances[i] < self.dis_th):
                filtered_dis.append(distances[i])
                src_indices.append(i)
                tar_indices.append(indices[i])

        # print("Total matches of ICP:",len(filtered_dis))
        return filtered_dis, src_indices, tar_indices

        '''
        Very slow, need to optimize
        '''
        # distances = []
        # src_indices = []
        # tar_indices = []
        # for i in range(src.shape[0]):
        #     min =  1000
        #     index = 0
        #     for j in range(tar.shape[0]):
        #         dist_tmp =  self.calcDist(src[i].tolist(), tar[j].tolist())
        #         if (dist_tmp < min):
        #             min = dist_tmp
        #             index = j
        #     if(min < self.dis_th):
        #         distances.append(min)
        #         src_indices.append(i)
        #         tar_indices.append(index)
        # return distances, src_indices, tar_indices

    def getTransform(self, src, tar):
        T = np.identity(3)
        A = src
        B = tar

        # get number of dimensions
        m = A.shape[1]

        # translate points to their centroids
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B

        # rotation matrix
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
            Vt[m-1,:] *= -1
            R = np.dot(Vt.T, U.T)

        # translation
        t = centroid_B.T - np.dot(R,centroid_A.T)

        # homogeneous transformation
        T = np.identity(m+1)
        T[:m, :m] = R
        T[:m, m] = t

        return T

    def calcDist(self, a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return math.hypot(dx, dy)