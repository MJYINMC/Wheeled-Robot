#!/usr/bin/env python
import math
import numpy as np

# f = open('/home/rosuser/catkin_ws/tmp/mapping.log', 'w')
def fprint(s):
    return
    f.write(s)
    f.write('\n')
    f.flush()

class Mapping():
    def __init__(self, xw, yw, xyreso):
        self.width_x = xw*xyreso
        self.width_y = yw*xyreso
        self.xyreso = xyreso
        self.xw = xw
        self.yw = yw
        # default 0.5 -- [[0.5 for i in range(yw)] for i in range(xw)]
        self.pmap = np.ones((self.xw, self.yw))/2
        self.minx = -self.width_x/2.0
        self.maxx = self.width_x/2.0
        self.miny = -self.width_y/2.0
        self.maxy = self.width_y/2.0

    def logodd(self, p):
        fprint('    logodd: p:{}'.format(p))
        if p == 0:
            return -2.94
        else: 
            return math.log(p / (1-p))

    def update_prob(self, old_p, new_p):
        fprint('  update_prob: old_p:{}, pn:{}'.format(old_p, new_p))
        el= math.exp(self.logodd(old_p) + self.logodd(new_p))
        return el/(1+el)

    def bresenham(self, p_ob, p_center):
        def is_in_map(point):
            x, y = point
            if 0 <= x < self.xw and 0 <= y < self.yw:
                return True
            return False

        res = []
        
        x0, y0 = p_ob
        x1, y1 = p_center
        x0, y0 = int(x0), int(y0)
        x1, y1 = int(x1), int(y1)
        steep = abs(y1 - y0) > abs(x1 - x0)
        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        dx = x1 - x0
        dy = y1 - y0
        derr = 2*abs(dy)
        err = 0

        ystep = 1 if y0 < y1 else -1
        x, y = x0, y0
        while x <= x1:
            if steep and is_in_map((y, x)):
                res.append((y, x))
            elif is_in_map((x, y)):
                res.append((x, y))

            err += derr
            if err > dx:
                y += ystep
                err -= 2* dx
            x += 1
        
        return res

    def update(self, ox, oy, center_x, center_y):
        def process_axis(a, b):
            return int(a/self.xyreso + b/2)

        center_x, center_y = process_axis(center_x, self.xw), process_axis(center_y, self.yw)
        for p_ob in zip(ox, oy):
            p_ob = (process_axis(p_ob[0], self.xw), process_axis(p_ob[1], self.yw))
            line = self.bresenham(p_ob, (center_x, center_y))
            for p_observed in line:
                prob = 0.8 if p_observed == p_ob else 0.1
                res = self.update_prob(self.pmap[p_observed], prob)
                if res > 0.95:
                    res = 0.95
                if res < 0.05:
                    res = 0.05
                self.pmap[p_observed] = 0 if math.isnan(res) else res

        return self.pmap
