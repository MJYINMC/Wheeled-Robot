from queue import Queue
import numpy as np

class Expander:
    def __init__(self, origmap, hmax, wmax, lost_temp):
        self.origmap = origmap
        self.newMap = origmap.reshape(hmax, wmax)
        self.hmax = hmax
        self.wmax = wmax
        self.lost_temp = lost_temp

    def getNeighbors(self, point, temp):
        neighbors = [(-1, 1), (0, 1), (1, 1), (-1, 0), (1,0), (-1, -1), (0, -1), (1, -1)]

        x = point[0]
        y = point[1]

        for neighbor in neighbors:
            neighbor_point = (x+neighbor[0], y+neighbor[1], temp)
            if (0 < neighbor_point[0] < self.hmax) and (0 < neighbor_point[1] < self.wmax):
                yield neighbor_point

    def getNewMap(self):
        TempQueue = Queue()
        for w in range(0,129):
            for h in range(0,129):
                if (self.origmap[w][h] > 50):
                    TempQueue.put_nowait((w, h, 100))
        while not TempQueue.empty():
            p = TempQueue.get_nowait()
            if self.newMap[(p[0], p[1])] > p[2] and p[2] != 100:
                continue
            self.newMap[(p[0], p[1])] = p[2]
            next_temp = p[2]-self.lost_temp
            if next_temp < 0:
                break
            for neighbor in self.getNeighbors(p, next_temp):
                TempQueue.put_nowait(neighbor)
        return self.newMap