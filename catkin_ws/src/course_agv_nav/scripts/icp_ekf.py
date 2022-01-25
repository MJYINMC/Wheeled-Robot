#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numpy.core.fromnumeric import ravel
from numpy.lib.type_check import real
import rospy
import tf
import math
from nav_msgs.srv import GetMap
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray,Marker
from nav_msgs.msg import OccupancyGrid
import numpy as np
import cv2
from icp import ICP
from ekf import EKF,STATE_SIZE
from extraction import Extraction
from sift_match import sift_match
import matplotlib.pyplot as plt

MAX_LASER_RANGE = 30

def r2d(r):
    return r*180/math.pi

class SLAM_EKF():
    def __init__(self):
        # ros param
        self.robot_x = rospy.get_param('/slam/robot_x',0)
        self.robot_y = rospy.get_param('/slam/robot_y',0)
        self.robot_theta = rospy.get_param('/slam/robot_theta',0)

        self.icp = ICP()
        self.ekf = EKF()
        self.extraction = Extraction()

        # odom robot init states
        self.sensor_sta = [self.robot_x,self.robot_y,self.robot_theta]
        self.isFirstScan = True
        self.src_pc = []
        self.tar_pc = []
        self.map = None

        # State Vector [x y yaw]
        self.xOdom = np.zeros((3, 1))
        self.xEst = np.zeros((3, 1))
        self.PEst = np.zeros((3, 3))
        
        # ros topic
        self.laser_sub = rospy.Subscriber('/scan',LaserScan,self.laserCallback)
        self.map_sub = rospy.Subscriber('/map',OccupancyGrid,self.mapCallback)
        self.location_pub = rospy.Publisher('ekf_location',Odometry,queue_size=3)
        self.odom_pub = rospy.Publisher('icp_odom',Odometry,queue_size=3)
        self.odom_broadcaster = tf.TransformBroadcaster()
        self.landMark_pub = rospy.Publisher('/landMarks',MarkerArray,queue_size=1)
        self.tf = tf.TransformListener()

        self.resolution = 0
        self.count = 0

    def get_real_loc(self):
        self.tf = tf.TransformListener()
        self.tf.waitForTransform("/base_footprint", "/map", rospy.Time(),
                                    rospy.Duration(4.0))
        l2m, rot = self.tf.lookupTransform('/base_footprint', '/map', rospy.Time(0))
        euler = tf.transformations.euler_from_quaternion(rot)
        roll, pitch, yaw_l2m = euler[0], euler[1], euler[2]

        self.tf.waitForTransform("/map", "/world_base", rospy.Time(),
                                    rospy.Duration(4.0))
        m2w, rot = self.tf.lookupTransform('/map', '/world_base', rospy.Time(0))
        euler = tf.transformations.euler_from_quaternion(rot)
        roll, pitch, yaw_m2w = euler[0], euler[1], euler[2]

        dx = -l2m[0] * math.cos(yaw_l2m) - l2m[1] * math.sin(yaw_l2m) - m2w[0]
        dy = l2m[0] * math.sin(yaw_l2m) - l2m[1] * math.cos(yaw_l2m) - m2w[1]
        dthela = -yaw_l2m
        z_world = np.array([    [dx * math.cos(yaw_m2w) + dy * math.sin(yaw_m2w)],
                                [- dx * math.sin(yaw_m2w) + dy * math.cos(yaw_m2w)],
                                [dthela - yaw_m2w]
                                ])
        return z_world

    def match(self, np_msg):
        self.count += 1
        print ("Match with static map!")
        real_loc =  self.get_real_loc()
        if(self.count <= 10):
            x, y, yaw = real_loc[0, 0], real_loc[1, 0], real_loc[2, 0]
        else:
            x, y, yaw = self.xEst[0, 0], self.xEst[1, 0], self.xEst[2, 0]
        # print ("Current position:", x, y ,yaw)

        x_min = np.min(np_msg[0, :])
        y_min = np.min(np_msg[1, :])
        x_max = np.max(np_msg[0, :])
        y_max = np.max(np_msg[1, :])

        # laser_frame = np.zeros((int((y_max-y_min)/self.resolution) + 5, int((x_max - x_min)/self.resolution) + 5),dtype=np.uint8)
        # laser_frame[:,:] = 255
        # print ("laser frame size:", laser_frame.shape)
        # for i in range(np_msg.shape[1]):
        #     y = int((np_msg[1, i] - y_min) / self.resolution) + 1
        #     x = int((np_msg[0, i] - x_min) / self.resolution) + 1
        #     laser_frame[y, x] = 0

        # cv2.imshow('test', laser_frame)
        # cv2.waitKey(0)
        # cv2.destroyWindow('test')
        # sift_match(cv2.cvtColor(self.map, cv2.COLOR_GRAY2RGB), cv2.cvtColor(laser_frame, cv2.COLOR_GRAY2RGB))

        # Useless method
        # T = self.icp.process(np_msg, self.map)
        # print (T)
        # print (self.T2u(T))
        # return self.T2u(T)

        

        # Another method
        src_pc = np_msg.copy()
        dx = src_pc[0, :]
        dy = src_pc[1, :]
        src_pc[0, :] = dx * math.cos(yaw) - dy * math.sin(yaw) + x
        src_pc[1, :] = dx * math.sin(yaw) + dy * math.cos(yaw) + y
        x_min = np.min(src_pc[0, :])
        y_min = np.min(src_pc[1, :])
        x_max = np.max(src_pc[0, :])
        y_max = np.max(src_pc[1, :])
        tmp_map = self.map[:, (self.map[0, :] > x_min) & (self.map[1, :] > y_min) 
                            & (self.map[0, :] < x_max) & (self.map[1, :] < y_max) ]

        # Show piece of the map
        # self.count += 1
        # if(self.count % 200 == 0):
        #     plt.axis("equal")
        #     plt.plot(tmp_map[0, :], tmp_map[1, :], marker='*', linestyle='')
        #     plt.show()
        #     plt.axis("equal")
        #     plt.plot(src_pc[0, :], src_pc[1, :], marker='*', linestyle='')
        #     plt.show()
        # return np.array([[0],[0],[0]]), 0

        try:
            T, matches = self.icp.process(src_pc, tmp_map)
        except:
            return np.array([[0],[0],[0]]), 0
        u = self.T2u(T)
        z = np.array([[x-u[0, 0]],[y-u[1, 0]],[yaw-u[2, 0]]])
        if(self.count <= 10):
            return real_loc, 1000000
        else:
            return z, matches
        # return real_loc, 1000000

    def laserCallback(self, msg):
        real_loc =  self.get_real_loc()
        print ("Real position:", real_loc[0, 0], real_loc[1, 0], real_loc[2, 0])
        np_msg = self.laserToNumpy(msg)
        # lm = self.extraction.process(np_msg)
        u = self.calc_odometry(np_msg)
        z, matches = self.match(np_msg)
        self.xEst, self.PEst = self.ekf.estimate(self.xEst, self.PEst, z, u, matches)
        # self.xEst, self.PEst = z, np.zeros((3,3))
        self.publishResult()
        pass

    def mapCallback(self, msg):
        print("Map received!!!")
        print ("reso:", msg.info.resolution)
        print ("Origin:", (msg.info.origin.position.x, msg.info.origin.position.y))
        self.tf = tf.TransformListener()
        self.tf.waitForTransform("/map", "/world_base", rospy.Time(),
                                    rospy.Duration(4.0))
        trans, rot = self.tf.lookupTransform('/map', '/world_base', rospy.Time(0))
        euler = tf.transformations.euler_from_quaternion(rot)
        roll, pitch, yaw = euler[0], euler[1], euler[2]
        print ("Trans:", trans)
        print ("Yaw:", yaw)
        print ("Map height:", msg.info.height, "Map width:", msg.info.width)
        self.map = np.array(msg.data).reshape(msg.info.height, msg.info.width).astype(np.uint8)
        lines = cv2.Canny(self.map, 50, 150, apertureSize = 3)
        rows, cols = lines.shape

        # M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0), r2d(yaw), 1)
        # lines = cv2.warpAffine(lines,M,(cols,rows))
        # lines = 255 - lines
        # self.resolution = msg.info.resolution
        # self.map = lines
        # cv2.imshow('Canny', lines)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # oy, ox = np.nonzero(lines > 0)



        # get critical points
        ox, oy = np.nonzero(lines > 0)
        ox = ox*msg.info.resolution +  msg.info.origin.position.y
        oy = oy*msg.info.resolution +  msg.info.origin.position.x

        # Show map base 
        # plt.axis("equal")
        # plt.plot(oy, ox, marker='*', linestyle='')
        # plt.show()

        ox = ox - trans[1]
        oy = oy - trans[0]

        world_base_x = oy * math.cos(yaw) + ox * math.sin(yaw)
        world_base_y = - oy * math.sin(yaw) + ox * math.cos(yaw)
        print ("Total points from static map:", ox.shape[0])
        self.map = np.zeros((3, ox.shape[0]))
        self.map[0, :] = world_base_x
        self.map[1, :] = world_base_y
        self.map[2, :] = 1

        # Show world base
        # plt.axis("equal")
        # plt.plot(ox, oy, marker='*', linestyle='')
        # plt.show()
        
    def observation(self, lm):
        landmarks = lm
        z = np.zeros((0, 3))
        for i in range(len(landmarks.id)):
            dx = landmarks.position_x[i]
            dy = landmarks.position_y[i]
            d = math.hypot(dx, dy)
            angle = self.ekf.pi_2_pi(math.atan2(dy, dx))
            zi = np.array([d, angle, i])
            z = np.vstack((z, zi))
        return z

    def calc_odometry(self, np_msg):
        if self.isFirstScan:
            self.tar_pc = np_msg
            self.isFirstScan = False
            return np.array([[0.0,0.0,0.0]]).T
        self.src_pc = np_msg
        transform_acc, matches = self.icp.process(self.tar_pc, self.src_pc)
        self.tar_pc = np_msg
        return self.T2u(transform_acc)

    def laserToNumpy(self,msg):
        total_num = len(msg.ranges)
        pc = np.ones([3,total_num])
        range_l = np.array(msg.ranges)
        range_l[range_l == np.inf] = MAX_LASER_RANGE
        angle_l = np.linspace(msg.angle_min,msg.angle_max,total_num)
        pc[0:2,:] = np.vstack((np.multiply(np.cos(angle_l),range_l),np.multiply(np.sin(angle_l),range_l)))
        pc = pc[:, ~np.any(np.isnan(pc), axis=0)]
        return pc

    def T2u(self, t):
        dw = math.atan2(t[1,0],t[0,0])
        u = np.array([[t[0,2],t[1,2],dw]])
        return u.T
    
    def u2T(self, u):
        w = u[2]
        dx = u[0]
        dy = u[1]

        return np.array([
            [ math.cos(w), -math.sin(w), dx],
            [ math.sin(w),  math.cos(w), dy],
            [0,0,1]
        ])

    def lm2pc(self, lm):
        total_num = len(lm.id)
        dy = lm.position_y
        dx = lm.position_x
        range_l = np.hypot(dy,dx)
        angle_l = np.arctan2(dy,dx)
        pc = np.ones((3,total_num))
        pc[0:2,:] = np.vstack((np.multiply(np.cos(angle_l),range_l),np.multiply(np.sin(angle_l),range_l)))
        pc = pc[:, ~np.any(np.isnan(pc), axis=0)]
        return pc

    def publishResult(self):
        # tf
        s = self.xEst.reshape(-1)
        q = tf.transformations.quaternion_from_euler(0,0,s[2])
        self.odom_broadcaster.sendTransform((s[0],s[1],0.001),(q[0],q[1],q[2],q[3]),
                            rospy.Time.now(),"ekf_location","world_base")
        # odom
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "world_base"

        odom.pose.pose.position.x = s[0]
        odom.pose.pose.position.y = s[1]
        odom.pose.pose.position.z = 0.001
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]

        self.location_pub.publish(odom)

        s = self.xOdom
        q = tf.transformations.quaternion_from_euler(0,0,s[2])
        # odom
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "world_base"

        odom.pose.pose.position.x = s[0]
        odom.pose.pose.position.y = s[1]
        odom.pose.pose.position.z = 0.001
        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]

        self.odom_pub.publish(odom)
        
        pass

def main():
    rospy.init_node('slam_node')
    s = SLAM_EKF()
    rospy.spin()

if __name__ == '__main__':
    main()