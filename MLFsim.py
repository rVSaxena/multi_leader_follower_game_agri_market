import numpy as np
from numpy.linalg import inv, det
from instanceSolver import instanceSolver
from globalSolver import globalSolver
import multiprocessing
import matplotlib.pyplot as plt
from scipy.stats import norm
import sys


if __name__=='__main__':
    numLeaders=8
    Q=np.vstack((-np.eye(numLeaders), np.ones((numLeaders,))))
    r=np.asarray([0]*numLeaders+[50])
    B=np.eye(numLeaders)
    mk_divisor=3.0
    a=2000.0
    b=1.0/7.0
    w=np.random.uniform(1.5,1.2,size=(numLeaders, ))
    mode=0
    if len(sys.argv)>1:
        mode=int(sys.argv[1])
    LeaderLow, LeaderHigh=np.asarray([0]*numLeaders), np.random.uniform(30,70,size=(numLeaders,))
    if mode!=0:
        follower_deficit_variance=np.random.randint(4,10)
        follower_deficit_mean=np.random.randint(1,5)
        leader_loss_limit=np.random.uniform(100,300,size=(numLeaders, ))
        leader_epsilons=np.random.uniform(0.0,0.1,size=(numLeaders, ))
    if mode==1:
        temp=norm.ppf(1-leader_epsilons)*np.sqrt(follower_deficit_variance)+follower_deficit_mean
        LeaderHigh=np.minimum(LeaderHigh, leader_loss_limit/temp)
    elif mode==2:
        temp=np.sqrt(np.divide(1-leader_epsilons, leader_epsilons)*follower_deficit_variance)+follower_deficit_mean
        LeaderHigh=np.minimum(LeaderHigh, leader_loss_limit/temp)
    solver=globalSolver(w,a,b,LeaderLow,LeaderHigh,mk_divisor,Q,r,B)
    p,y,qq,qqqq=solver.solve(numProcessses=10)
    print("The prices are {} and the quantities are {}".format(p,y))