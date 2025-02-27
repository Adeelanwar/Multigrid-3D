# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 13:17:48 2019

@author: Adeel
"""
import time
from Project3D import *
from jacobi import jac
from rbjac import rbjac
from guass import guass
import scipy.sparse.linalg as LA2
from numpy import linalg as LA
def twogrid(grid,domain,Ah, rhs, uh):
    #step 0
    if(grid[0] ==2 or grid [1] == 2 or grid[2] == 2):
        print("Sorry grid is already coarse")
        return uh
    
    #step 1 Presmooting
    uh = guass(Ah, rhs, uh, 1)
#    uh = rbjac(Ah,rhs,uh,1,1,grid)
    #step 2 
    
    rh = rhs - Ah @ uh #residual calculation on fine grid
    R = rest3d(grid)
    P = prol3d(grid)
    rH = R @ rh
    N = np.size(rH)
    AH = Mat3D((grid / 2).astype(int), domain)
    eH = np.reshape(LA2.spsolve(AH, rH),(N, 1))
    eh = P @ eH
    uh = uh + eh
    
    #step 3
    uh = guass(Ah, rhs, uh, 1)
    
    return uh
