import warnings
import multiprocessing
import numpy as np
from numpy.linalg import inv

class instanceSolver:
    """
    Class to solve a MLF equilibrium assuming a particular set of follower constraints as active.
    """
    def __init__(self, w, a, b, LeaderLow, LeaderHigh, mk_divisor, **kwargs):
        """
        Initializer.
        Arguments:
            w: Leader processing cost vector.
            a, b: inverse demand function coefficients, p(q)=a-bq.
            LeaderLow, LeaderHigh: The vectors capturing the costraints  p>=low and p<=high, respectively.
            mk_divisor: The market price divisor for the government.
            kwargs: Supply some/all of "unconstrained", "processing_matrix", "equality_vector" and "constraints_matrix".
        """
        self.w=w
        self.a=a
        self.b=b
        self.LeaderLow=LeaderLow
        self.LeaderHigh=LeaderHigh
        self.mk_divisor=mk_divisor
        self.unconstrained=kwargs.get("unconstrained", False)
        self.A1=kwargs.get("constraints_matrix").reshape((-1,len(w)))
        self.b1=kwargs.get("equality_vector")
        self.B=kwargs.get("processing_matrix")

        self.yC_, self.yD_=self.follower_solver()
        for i in range(len(self.yD_)):
            if self.yD_[i,i]*(2*self.b*np.sum(self.yD_[:,i])+2+self.w[i]*self.yD_[i,i])<0:
                warnStr="Non Convex at i={} with value {}".format(i, self.yD_[i,i]*(2*self.b*np.sum(self.yD_[:,i])+2+self.w[i]*self.yD_[i,i]))
                warnings.warn(warnStr)
        return

    def follower_solver(self):
        """
        Solves the follower problem. Is sufficient to run this just once.
        Returns:
            C, D such that y=C+Dp
        """

        if self.unconstrained:
            return np.zeros((self.B.shape[0], )), inv(self.B)
        
        B_inv=inv(self.B)
        return B_inv@self.A1.T@inv(self.A1@B_inv@self.A1.T)@self.b1, B_inv-B_inv@self.A1.T@inv(self.A1@B_inv@self.A1.T)@self.A1@B_inv

    def leaderGradientHelper(self, arg):
        """
        Just a helper function to compute gradients simultaneously on all CPU cores.
        Computes the gradient of the ith leader at the current state.
        """
        i=arg[0]
        y=self.yC_+self.yD_@arg[1]
        p=arg[1]
        k=arg[2]
        y_sum=np.sum(y)
        return y[i]+p[i]*self.yD_[i,i]+self.w[i]*y[i]*self.yD_[i,i]-(self.a-self.b*y_sum)*self.yD_[i,i]/k+y[i]*self.b*np.sum(self.yD_[:, i])/k

    def leaderCostGradient(self, p):
        """
        Arguments:
            p: current price vector
        Returns:
            The Leader Cost Gradient
        """
        
        numLeaders=len(p)
        helperArguments=[(i, p, 1) for i in range(numLeaders-1)]
        helperArguments.append((numLeaders-1, p, self.mk_divisor))

        leaderGradient=np.fromiter(map(self.leaderGradientHelper, helperArguments), dtype=np.float32)

        return leaderGradient

    def g(self, x):
        """
        The 'activation' function, of sorts.
        """
        return np.min((self.LeaderHigh, np.max((x, self.LeaderLow), axis=0)), axis=0)

    def train_loop(self, **kwargs):
        """
        Solves the MLF-Game for the objects' particular selection of active follower constraints.
        Arguments:
            kwargs: Optionally, supply tolerance and learningRate
        Returns:
            The Leader-Follower Nash equilibrium
        """
        numLeaders=len(self.w)
        tolerance=kwargs.get("tolerance", 1e-5)
        learningRate=kwargs.get("learningRate", 5*1e-3)


        p=np.random.uniform(self.LeaderLow, self.LeaderHigh)
        change=float('inf')
        while(change>tolerance):
            leaderGrad=self.leaderCostGradient(p)
            delta=learningRate*(self.g(p-leaderGrad)-p)
            change=np.max(np.abs(delta))
            p=p+delta
        
        return p, self.yC_+self.yD_@p