import math
import numpy as np
from numpy import ma


M_DIST_TH = 0.6  # Threshold of Mahalanobis distance for data association.
STATE_SIZE = 3  # State size [x,y,yaw]
LM_SIZE = 2  # LM state size [x,y]

############################################################### 
################## System description ########################
############################################################### 
# x_k = f(x_{k-1}, u_k) + w_k
# y_k = h(x_k) + v_k
# where h() and f() are nonlinear function
# x_{k-1}, x_k are the state vector at timestamp k-1 and k
# w_k and v_k are system noise and observation noise
# y_k is the observation at timestamp k
# Notation: Jf(x) and Jh(x) are the Jacobian matrix of f(x) and h(x) 
#######################################################################

# EKF system noise density: Qx = E[w_k * w_k']
# Qx = np.diag([0.35, 0.35, np.deg2rad(15.0)]) ** 2

# Qx = np.diag([1/np.log2(1.00001), 1/np.log2(1.00001), 1/np.log2(1.00001)]) ** 2
# Qx = np.diag([0.35, 0.35, np.deg2rad(15.0)]) ** 2

u_match = 500
u_variance = np.array([
    [1/np.log2(1.00001 + u_match), 0.0, 0.0],
    [0.0, 1/np.log2(1.00001 + u_match), 0.0],
    [0.0, 0.0, 0.5/np.log2(1.00001 + u_match)]
    ])

class EKF():
    def __init__(self):
        pass
    
    # input: posteriori estimated state and its covariance at timestamp k, system input u
    # output: posteriori estimated state and its covariance at timestamp k+1
    def estimate(self, xEst, PEst, z, u, matches):
        # laser odom
        x_odom = self.odom_model(xEst, u)  # 3,1
        g1 = self.jacob_1(xEst, u)
        g2 = self.jacob_2(xEst, u)
        variance1 = np.dot(np.dot(np.dot(g1, PEst), PEst.T), g1.T)              # 3,3
        variance2 = np.dot(np.dot(np.dot(g2, u_variance), u_variance.T), g2.T)  # 3,3
        sigma1 = variance1 + variance2                                          # 3,3
        sigma2 = np.diag([1/np.log2(1.0000001 + matches), 1/np.log2(1.0000001 + matches), 1/np.log2(1.0000001 + matches)]) ** 2
        # print ("sigma1", sigma1)
        # print ("Qx", Qx)                                                        
        x = np.zeros((3, 1))
        P = np.zeros((3, 3))
        x = np.dot(np.linalg.inv(np.linalg.inv(sigma1) + np.linalg.inv(sigma2)), (np.dot(np.linalg.inv(sigma1), x_odom) + np.dot(np.linalg.inv(sigma2), z)))
        P = sigma1 - np.dot(np.dot(sigma1, np.linalg.inv(sigma1 + sigma2)), sigma2)
        # print ("P:", P)
        return x, P

    # input: posteriori estimated state at timestamp k, system input u
    # output: priori estimated state at timestamp k+1 
    def odom_model(self, xEst, u):
        """
            x = [x,y,w,...,xi,yi,...]T
            u = [dx,dy,dyaw]        prev_error = 0
        """
        x = np.zeros((3,1))
        yaw = xEst[2,0]
        dx = u[0,0]
        dy = u[1,0]
        dyaw = u[2,0]
        x[0,0] = xEst[0,0] + math.cos(yaw)*dx - math.sin(yaw)*dy 
        x[1,0] = xEst[1,0] + math.sin(yaw)*dx + math.cos(yaw)*dy 
        x[2,0] = xEst[2,0] + dyaw
        return x

    # g1
    def jacob_1(self, x, u):
        """
        Jacobian of Odom Model
        x = [x,y,w,...,xi,yi,...]T
        u = [ox,oy,ow]T
        """
        yaw = x[2, 0]
        dx = u[0, 0]
        dy = u[1, 0]
        # g1 = np.array(
        #             [
        #             [math.cos(dyaw) , math.sin(dyaw), 0.0],
        #             [- math.sin(dyaw) , math.cos(dyaw), 0.0],
        #             [0.0 , 0.0, 1.0]
        #             ])
        g1 = np.array(
                    [
                    [1.0, 0.0,  -dx*math.sin(yaw) - dy*math.cos(yaw)],
                    [0.0, 1.0, dx*math.cos(yaw) - dy*math.sin(yaw)],
                    [0.0, 0.0, 1.0]
                    ])
        return g1

    # g2
    def jacob_2(self, x, u):
        x_t = x[0, 0]
        y_t = x[1, 0]
        yaw = x[2, 0]
        dx = u[0, 0]
        dy = u[1, 0]
        dyaw = u[2, 0]
        # g2 = np.array(
        #             [
        #             [-math.cos(dyaw), -math.sin(dyaw), -math.sin(dyaw)*x_t + math.cos(dyaw)*y_t + math.sin(dyaw)*dx - math.cos(dyaw)*dy],
        #             [math.sin(dyaw), math.cos(dyaw), -math.cos(dyaw)*x_t - math.sin(dyaw)*y_t + math.cos(dyaw)*dx + math.sin(dyaw)*dy],
        #             [0, 0, 1]
        #             ])
        g2 = np.array(
                    [
                    [math.cos(yaw), -math.sin(yaw), 0],
                    [math.sin(yaw), math.cos(yaw), 0],
                    [0, 0, 1]
                    ])
        return g2