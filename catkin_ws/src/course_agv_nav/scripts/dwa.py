#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import math
import time
import matplotlib.pyplot as plt

PI = math.pi

class DWAPlanner:
    def __init__(self):
        self.max_speed =  0.12 # [m/s]
        self.min_speed =  0 # [m/s]
        self.max_yawrate = 0.15  # [rad/s]
        
        self.predict_time = 5  # [s]
        self.dt = 0.1              # [s] Time tick for motion prediction

        self.max_accel = 0.5     # [m/ss]
        self.max_dyawrate = 180.0 * PI / 180.0  # [rad/ss]

        # 窗口分辨率，线速度和角速度分辨率
        self.v_reso = 0.02       # [m/s]
        self.yawrate_reso = 5 * PI / 180.0   # [rad/s]

        # heading cost 轨迹末车头朝向与终点连线之间的偏角，越小越好
        self.heading_weight = 0
        # dist cost 轨迹中距离障碍物的距离，越大越好，使用1/min_dist转换为越小越好便于计算
        self.dist_weight = 20
        # vel cost 车的速度，越大越好，时间花费短，使用max_speed - vel转换为越小越好便于计算
        self.vel_weight = 5
        self.goal_weight = 20
        # 车的半径
        self.radius = 0.15

    def plan(self, x, goal, ob):
        # 获取本次规划的线速度和角速度窗口，受到加速度和最大速度的制约
        dw = self.calc_window_range(x)
        # print "State:", x
        # print "Goal:", goal
        # print "Dynamic Window:", dw
        # 计算最佳的角速度和线速度
        u = self.calc_vx_vw(dw, x, goal, ob)
        return u

    def calc_window_range(self, x):
        """
        calculation dynamic window based on current state x
        x = [x, y, yaw, vx, vw]
        """
        # Dynamic window from robot specification
        Vs = [self.min_speed, self.max_speed,
            -self.max_yawrate, self.max_yawrate]

        # Dynamic window from motion model
        Vd = [x[3] - self.max_accel * self.dt,
            x[3] + self.max_accel * self.dt,
            x[4] - self.max_dyawrate * self.dt,
            x[4] + self.max_dyawrate * self.dt]

        #  [vmin, vmax, yaw_rate min, yaw_rate max]
        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
            max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
        return dw
    
    def calc_vx_vw(self, dw, x, goal, ob):
        min_cost = float("inf")
        best_u = (0, 0)
        heading_total = 0
        dist_total = 1
        vel_total = 0
        goal_total = 0
        table = {}
        # 记录下每一个vx vw对应的h_cost, d_cost, v_cost
        for vx in np.arange(dw[0], dw[1], self.v_reso):
            for vw in np.arange(dw[2], dw[3], self.yawrate_reso):
                traj = self.predict_trajectory(vx, vw)
                h_cost = self.calc_heading(traj, goal)
                heading_total += h_cost
                
                d_cost = self.calc_dist(traj, ob)
                dist_total += d_cost
                
                v_cost = self.calc_vel(vx)
                vel_total += v_cost
                
                g_cost = self.calc_dis(traj, goal)
                goal_total += g_cost

                table[(vx, vw)] = (h_cost, d_cost, v_cost, g_cost)
        
        # print(table)
        # 查找最优的vx和vw
        for candidate in table:
            # 如果该路径会撞上障碍物，则无需考虑这条路径
            if(table[candidate][1] == 0):
                continue;
            # 归一化处理
            normaled_h = self.heading_weight * table[candidate][0] / heading_total
            normaled_d = self.dist_weight * table[candidate][1] / dist_total
            normaled_v = self.vel_weight * table[candidate][2] / vel_total
            normaled_g = self.goal_weight * table[candidate][3] / goal_total
            total_cost = normaled_h + normaled_d + normaled_v + normaled_g
            if( total_cost < min_cost):
                min_cost = total_cost
                best_u = candidate
        return best_u

    def calc_heading(self, traj, goal):
        dx = goal[0] - traj[-1][0] 
        dy = goal[1] - traj[-1][1] 
        # print "dx, dy:", dx, dy
        error_angle = math.atan2(dy, dx)
        # print "angle", error_angle
        cost_angle = error_angle - traj[-1][2]
        dis = math.hypot(dx,dy)
        cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle))) 
        return cost

    def calc_dis(self, traj, goal):
        dx = goal[0] - traj[-1][0] 
        dy = goal[1] - traj[-1][1]
        dis = math.hypot(dx, dy)
        return dis

    def calc_dist(self, traj, ob):
        """
        如果与障碍物碰撞，则返回0
        """
        min_dist = float("inf")

        # 求路径上的采样点与障碍物之间的最小距离
        for sample in traj:
            nearest = float("inf")
            for i in range(len(ob)):
                dist = math.hypot(sample[0] - ob[i,0], sample[1] - ob[i,1])
                if(dist < nearest):
                    nearest = dist
            if(nearest < min_dist):
                min_dist =  nearest

        # for i in range(len(ob)):
        #     dist = math.hypot(traj[-1][0] - ob[i,0], traj[-1][1] - ob[i,1])
        #     if dist < min_dist:
        #         min_dist = dist

        if(min_dist < self.radius):
            return 0
        return  1/min_dist

    def calc_vel(self, vx):
        return (self.max_speed - abs(vx))

    def predict_trajectory(self, vx, vw):
        """
        predict trajectory with an input
        x = [x, y, yaw, vx, vw]
        """
        x =  (0, 0, 0)
        traj = []
        time = 0
        while time <= self.predict_time:
            x = self.next_pose(x, vx, vw)
            traj.append(x)
            time += self.dt
        return traj

    def next_pose(self, x, vx, vw):
        next = (x[0] + vx * math.cos(x[2]) * self.dt, x[1] + vx * math.sin(x[2]) * self.dt, x[2] + vw * self.dt)
        return next