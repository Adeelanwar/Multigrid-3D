# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:05:39 2019

@author: Adeel
"""
import time
from Project3D import *
from jacobi import jac
from rbjac import rbjac
from guass import Gauss_Seidel
from guass import guass
from scipy.sparse import linalg as LA2
from numpy import linalg as LA
import scipy as sp
def multigrid(grid,domain,Ah, rhs, uh, cycleindex):
    #step 0
    if(grid[0] ==2 or grid [1] == 2 or grid[2] == 2):
        print("Sorry grid is already course")
        return uh
    
    #step 1 Presmooting
#    uh = Gauss_Seidel(Ah, rhs, uh, 1)
    uh = guass(Ah, rhs, uh, 1)
#    uh = jac(Ah, rhs, uh, 2, 6/7)
    uh = rbjac(Ah,rhs,uh,1,1,grid)
    #step 2 

    rh = rhs - Ah @ uh #residual calculation on fine grid
    R = rest3d(grid)
    P = prol3d(grid)
    grid = (grid / 2).astype(int)
    rH = R @ rh
    N = np.size(rH)
    AH = Mat3D(grid, domain)
    if(grid[0] == 2 or grid [1] == 2 or grid[2] == 2):
        eH = np.reshape(LA2.spsolve(AH, rH),(N, 1))
    else:
        eH = sp.zeros(((grid[0] - 1) * (grid[1] - 1) * (grid[2] - 1),1))
        for i in range(cycleindex):
            eH = multigrid(grid,domain,AH, rH, eH, cycleindex);
    eh = P @ eH
    uh = uh + eh
    
    grid = (grid * 2).astype(int)
    #step 3
#    uh = guass(Ah, rhs, uh, 1)
#    uh = Gauss_Seidel(Ah, rhs, uh, 1)
#    uh = jac(Ah, rhs, uh, 2, 6/7)
    uh = rbjac(Ah,rhs,uh,1,1,grid)
    return uh
