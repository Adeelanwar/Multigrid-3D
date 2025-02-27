# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 16:05:39 2019

@author: Adeel
"""
from scipy.sparse import diags
import numpy as np
from scipy.sparse import linalg as LA2
import scipy.sparse as sps

#grid is tuple that contains 3 tuples
def Mat3D(grid, domain):
    nx = grid[0] - 1
    ny = grid[1] - 1
    nz = grid[2] - 1 
    hx = (domain[0][1] - domain[0][0])/ (nx + 1)
    hy = (domain[1][1] - domain[1][0])/ (ny + 1)
    hz = (domain[2][1] - domain[2][0])/ (nz + 1)
#    stencil = [-1 / hz**2,-1 / hy**2,-1 / hx**2,(2 / hx**2) + (2 / hy**2) + (2 / hz**2),-1 / hx**2,-1 / hy**2,-1 / hz**2]
#    daig = [-1 * (nx * ny),-1 * (nx) ,-1,0,1,(nx),(nx * ny)]
#    dim = (nx) * (ny) * (nz)
#    A = (diags(stencil,daig,shape=((dim, dim))))
    
    A1d_x = (1/(hx**2)) * diags([-1,2,-1],[-1,0,1],(nx,nx))
    Ax = sps.kron(sps.identity(ny * nz), A1d_x,format = 'csr')

    A1d_y = (1/(hy**2)) * diags([-1,2,-1],[-1,0,1],(ny,ny));
    Ay = sps.kron(sps.identity(nz),sps.kron(A1d_y,sps.identity(nx)),format = 'csr')
    
    A1d_z = (1/(hz**2)) * diags([-1,2,-1],[-1,0,1],(nz,nz))
    Az = sps.kron(A1d_z,sps.identity(ny * nz), format = 'csr')

    return Ax + Ay + Az

def Mat2D(grid):
    nx = grid[0] - 1
    ny = grid[1] - 1 
    hx = (domain[0][1] - domain[0][0])/ (nx + 1)
    hy = (domain[1][1] - domain[1][0])/ (ny + 1)
    
    A1d_x = (1/(hx**2)) * diags([-1,2,-1],[-1,0,1],(nx,nx),format = 'csr')
    Ax = sps.kron(sps.identity(ny), A1d_x,format = 'csr');

    A1d_y = (1/(hy**2)) * diags([-1,2,-1],[-1,0,1],(ny,ny),format = 'csr')
    Ay = sps.kron(A1d_y,sps.identity(nx),format = 'csr');
    return Ax + Ay


#gridspacing tuple containing three elements number of spacing in x,y,z respectively.
def rest2d(gridspace):
    width = gridspace[0] - 1
    height = gridspace[1] - 1
    dim = width * height
    gridspace2 = (int(gridspace[0] / 2), int(gridspace[1] / 2))
    dim2 = int(((gridspace2[0] - 1) * (gridspace2[1] - 1)))
    I = sps.lil_matrix((dim2, dim))
    row = 0
    for j in range(gridspace2[1] - 1):
        for i in range(gridspace2[0] - 1):
            #row = i + j * (gridspace2[0] - 1)
            column = (2 * j + 1) * width + 2 * (i + 1) - 1
            I[row, column] = 4
            I[row, column - 1] = 2
            I[row, column + 1] = 2
            I[row, column - width] = 2
            I[row, column + width] = 2
            I[row, column - 1 - width] = 1
            I[row, column + 1 - width] = 1
            I[row, column - 1 + width] = 1
            I[row, column + 1 + width] = 1
            row += 1
    return I / 16

def prol2d(gridspace):
    gridspace2 = (gridspace[0] * 2, gridspace[1] * 2)
    I = rest2d(gridspace2)
    return 4 * I.transpose()

def rest3d(gridspace):
    width = gridspace[0] - 1
    height = gridspace[1] - 1
    depth = gridspace[2] - 1
    dim = int(width * height * depth)
    gridspace2 = (int(gridspace[0] / 2), int(gridspace[1] / 2),int(gridspace[2] / 2))
    dim2 = int(((gridspace2[0] - 1) * (gridspace2[1] - 1) * (gridspace2[2] - 1)))
    row = 0
    I = sps.lil_matrix((dim2, dim))
    for k in range(gridspace2[2] - 1):
        for j in range(gridspace2[1] - 1):
            for i in range(gridspace2[0] - 1):
                #row = i + j * (gridspace2[0] - 1) + k * ((gridspace2[1] - 1) * (gridspace2[0]  - 1))
                column = (2 * j + 1) * width + 2 * (i + 1) - 1 + (2 * k + 1) * (width * height)
                #middle level
                I[row, column] = 8
                I[row, column - 1] = 4
                I[row, column + 1] = 4
                I[row, column - width] = 4
                I[row, column + width] = 4
                I[row, column - 1 - width] = 2
                I[row, column + 1 - width] = 2
                I[row, column - 1 + width] = 2
                I[row, column + 1 + width] = 2
                #deep level
                column = (2 * j + 1) * width + 2 * (i + 1) - 1 + (2 * k + 2) * (width * height)
                I[row, column] = 4
                I[row, column - 1] = 2
                I[row, column + 1] = 2
                I[row, column - width] = 2
                I[row, column + width] = 2
                I[row, column - 1 - width] = 1
                I[row, column + 1 - width] = 1
                I[row, column - 1 + width] = 1
                I[row, column + 1 + width] = 1
                #upper level
                column = (2 * j + 1) * width + 2 * (i + 1) - 1 + (2 * k) * (width * height)
                I[row, column] = 4
                I[row, column - 1] = 2
                I[row, column + 1] = 2
                I[row, column - width] = 2
                I[row, column + width] = 2
                I[row, column - 1 - width] = 1
                I[row, column + 1 - width] = 1
                I[row, column - 1 + width] = 1
                I[row, column + 1 + width] = 1
                row += 1
    return I / 64

def prol3d(gridspace):
#    gridspace2 = (gridspace[0] * 2, gridspace[1] * 2, gridspace[2] * 2)
    I = rest3d(gridspace)
    return  8 * I.transpose()
