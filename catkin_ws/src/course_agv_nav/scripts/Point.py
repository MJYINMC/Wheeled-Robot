class Point:
    def __init__(self, x, y, parent, cost):
        self.x = x
        self.y = y
        self.parent = parent
        if(parent != None):
            self.BaseCost = cost + parent.BaseCost
        else:
            self.BaseCost = cost
