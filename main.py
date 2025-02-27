# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 00:30:18 2019

@author: Adeel
"""
import time
from functions.Project3D import *
from functions.Boundary import *
from functions.jacobi import *
from functions.twogrid import *
from functions.multigrid import multigrid
from numpy import linalg as LA
import scipy as sp
from functions.Boundary import source
import numpy as np


grid = np.array([16, 16, 16],dtype = int)
domain = ((0,1),(0,1),(0,1))

def iguess(grid):
    nx = (grid[0] - 1)
    ny = (grid[1] - 1)
    nz = (grid[2] - 1)
    return np.zeros((nx * ny * nz, 1))



def main(grid , domain, tol = 10**-6):
    Ah = Mat3D(grid, domain)
    x,y,z = interiorpoints(grid, domain)
    rhs = source(x, y, z) + boundary(grid, domain)
    uh = iguess(grid)
    rh = []
    mg_conv = []
    ua = analytic(x,y,z)
#    ui = LA2.spsolve(Ah.todense(),rhs)
    x = rhs - Ah @ uh
    rh.append(LA.norm(x,np.inf))
    for i in range(1000):
        uh = multigrid(grid,domain , Ah, rhs, uh, 1)
#        uh = twogrid(grid, domain , Ah, rhs, uh)
        x = rhs - Ah @ uh
        rh.append(LA.norm(x,np.inf))
        mg_conv.append(rh[i + 1] / rh [i])
        relres = rh[i + 1] / rh[0]
        print(i + 1, relres)
        if (relres <= tol):
            print(i + 1, relres)
            break
    return uh

start = time.time()
main(grid, domain)
end = time.time()
print("time taken: ", end - start)

