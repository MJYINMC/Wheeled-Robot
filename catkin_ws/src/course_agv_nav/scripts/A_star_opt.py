#!/usr/bin/env python
import math
import numpy as np
import time
from Point import Point 
from expander import Expander
import matplotlib.pyplot as plt

def cal_time(func):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print("%s running time: %s secs." % (func.__name__, t2-t1))
        # plt.show()
        return result
    return wrapper

offset = math.sqrt(2)

class Planner(object):
    def __init__(self, radius, map_data, origin_x, origin_y, width, height, resolution):
        print("Inited!")
        self.radius = radius
        self.map_data = map_data.copy()
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.width = width
        self.height = height
        self.resolution = resolution
        self.max_x = width * resolution + origin_x
        self.max_y = height * resolution + origin_y
        # neibor = radius/resolution        
        # for y in range(int(math.ceil(neibor)), height):
        #     for x in range(int(math.ceil(neibor)), width):
        #         if(map_data[x][y] > 50):
        #             self.map_data[int(math.floor(x-neibor)): int(math.ceil(x+neibor)), int(math.floor(y-neibor)): int(math.ceil(y+neibor))] = 100
        # ox,oy = np.nonzero(self.map_data > 50)
        # self.plan_ox = (ox).tolist()
        # self.plan_oy = (oy).tolist()
        # plt.axis("equal")
        # plt.plot(self.plan_ox, self.plan_oy,marker='*',linestyle='')
        # plt.show()
        self.expender = Expander(map_data, height, width, 30)
        self.map_data =  self.expender.getNewMap()      

        # ox,oy = np.nonzero(self.map_data == 100)
        # self.plan_ox = (ox).tolist()
        # self.plan_oy = (oy).tolist()
        # plt.axis("equal")
        # plt.plot(self.plan_ox, self.plan_oy,marker='*',linestyle='')
        # listx = []
        # listy = []
        # for i in range(0,129):
        #     for j in range(0,129):
        #         if(self.map_data[i,j] < 100 and self.map_data[i,j] >= 0):
        #             listx.append(i)
        #             listy.append(j)
        # plt.plot(listx, listy, marker='*',color = 'red',linestyle='')
        # plt.show()

    @cal_time
    def planning(self, sx, sy ,gx ,gy):
        print("Planning!")
        self.gx = gx;
        self.gy = gy;
        self.rx = []
        self.ry = []
        self.open_list = {}
        self.close_list = {}
        self.AStar(sx, sy)
        # self.rx.reverse()
        # self.ry.reverse()
        return self.rx, self.ry

    
    def AStar(self, sx, sy):
        ini_point = Point(sx, sy, None, 0)
        init_idx = self.CalcIndex(ini_point)
        self.open_list[init_idx] = ini_point
        while True:
            if(len(self.open_list)== 0):
                print("Empty list! No path found!")
                break
            
            index = self.GetNextPoint()
            next_point = self.open_list[index]
            # X, Y = self.CalcXYIdx(next_point)
            # plt.plot(X, Y, marker='*',color = 'yellow',linestyle='')
            if(self.IsEndPoint(next_point)):
                print("Algorithm Finished!")
                self.FetchPath(next_point)
                break
            self.close_list[index] = self.open_list[index]
            del self.open_list[index]

            self.ProbePoint(Point(next_point.x + self.resolution, next_point.y, 
                next_point, 1))
            self.ProbePoint(Point(next_point.x + self.resolution, next_point.y - self.resolution, 
                next_point, offset))
            self.ProbePoint(Point(next_point.x + self.resolution, next_point.y + self.resolution, 
                next_point, offset))
            self.ProbePoint(Point(next_point.x - self.resolution, next_point.y, 
                next_point, 1))
            self.ProbePoint(Point(next_point.x - self.resolution, next_point.y - self.resolution, 
                next_point, offset))
            self.ProbePoint(Point(next_point.x - self.resolution, next_point.y + self.resolution, 
                next_point, offset))
            self.ProbePoint(Point(next_point.x, next_point.y + self.resolution, 
                next_point, 1))
            self.ProbePoint(Point(next_point.x, next_point.y - self.resolution , 
                next_point, 1))

    def FetchPath(self, end_point):
        while end_point.parent != None:
            self.rx.append(end_point.x)
            self.ry.append(end_point.y)
            # X, Y = self.CalcXYIdx(end_point)
            # plt.plot(X, Y, marker='*',color = 'green',linestyle='')
            end_point = end_point.parent

    def ProbePoint(self, point):
        if self.CollisionDetection(point):
            return
        index = self.CalcIndex(point)
        if self.IsInCloseList(index):
            return
        if not self.IsInOpenList(index):
            self.open_list[index] = point
        elif(point.BaseCost < self.open_list[index].BaseCost):
            self.open_list[index].BaseCost = point.BaseCost

    def GetNextPoint(self):
        index = min(
            self.open_list,
            key = lambda o: self.open_list[o].BaseCost +
            self.HeuristicCost(self.open_list[o])
        )
        return index

    def CollisionDetection(self, point):
        if (point.x < self.origin_x 
            or point.x > self.max_x
            or point.y < self.origin_y
            or point.y > self.max_y):
            return True
        x_idx, y_idx = self.CalcXYIdx(point)
        if(self.map_data[x_idx][y_idx] > 0):
            return True
        return False
    
    def CalcIndex(self, point):
        return  int(
                (point.y - self.origin_y)/self.resolution * self.width  
                + (point.x - self.origin_x)/self.resolution)
    
    def CalcXYIdx(self, point):
        return  int((point.x - self.origin_x)/self.resolution), int((point.y - self.origin_y)/self.resolution)
    
    def HeuristicCost(self, point):
        # weight = 0.8/self.resolution
        weight = 1
        return weight * math.hypot(point.x - self.gx, point.y - self.gy)

    def IsInOpenList(self, index):
        return index in self.open_list

    def IsInCloseList(self, index):
        return index in self.close_list

    def IsEndPoint(self, point):
        return abs(point.x - self.gx) < self.resolution and abs(point.y - self.gy) < self.resolution