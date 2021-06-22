import warnings
import multiprocessing
import numpy as np
from numpy.linalg import inv, det
from instanceSolver import instanceSolver
from itertools import chain, combinations

class globalSolver:
    def __init__(self, w, a, b, LeaderLow, LeaderHigh, mk_divisor, Q, r, B):
        """
        Initializer.
        Arguments:
            w: Leader processing cost vector.
            a, b: inverse demand function coefficients, p(q)=a-bq.
            LeaderLow, LeaderHigh: The vectors capturing the costraints  p>=low and p<=high, respectively.
            mk_divisor: The market price divisor for the government.
            Q, r: The matrix and vector corresponding to the follower constraints, which are Qy<=r.
            B: The follower processing cost matrix
        """
        self.w=w
        self.a=a
        self.b=b
        self.LeaderLow=LeaderLow
        self.LeaderHigh=LeaderHigh
        self.mk_divisor=mk_divisor
        self.Q=Q
        self.r=r
        self.B=B
        return

    @staticmethod
    def powerset(iterable):
        s=list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    def solveInstance(self, activeConstraints):
        """
        Create an object of instanceSolver and solve the problem, where the active
        follower constraints are given by position where the bitmap is True.
        If the A1 formed is not full rank, we simply skip it as we are searching with brute force,
        hence all subsets, and therefore all subsets that are LI, will be considered.
        Arguments:
            activeConstraints: A list containing the index of active constraints.
        Returns:
            p, y: The MLF Nash Equilibrium
        """
        if len(activeConstraints)>0 and det(self.Q[activeConstraints,:]@self.B@self.Q[activeConstraints, :].T)<=1e-6:
            return (np.zeros((len(self.w),)), np.zeros((len(self.w),)), False, float('inf'))
        insSolver=instanceSolver(self.w, self.a, self.b, self.LeaderLow, self.LeaderHigh, self.mk_divisor,
            unconstrained=len(activeConstraints)==0, processing_matrix=self.B, constraints_matrix=self.Q[activeConstraints,:], equality_vector=self.r[activeConstraints,])
        p, y=insSolver.train_loop()
        admissible=(self.Q@y<=self.r).all()
        followerCost=-p.T@y+y.T@self.B@y/2
        return (p, y, admissible, followerCost)

    def solve(self, **kwargs):
        """
        Solves the problem by searching over the space of active constraints.
        Arguments:
            kwargs: Optionally supply number of processes.
        Returns:
            The MLF Nash equilibrium(s).
        """
        numConstraints=self.Q.shape[0]
        powSet=globalSolver.powerset(range(numConstraints))
        numProcesses=kwargs.get("numProcesses", 10)

        proc=multiprocessing.Pool(processes=numProcesses)
        results=proc.map(self.solveInstance, powSet)
        proc.close()
        proc.join()

        finResult=min(results, key=lambda x:x[3] if x[2] else float('inf'))

        return finResult






