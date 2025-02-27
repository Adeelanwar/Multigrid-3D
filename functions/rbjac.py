# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 09:12:15 2019

@author: Adeel
"""
import numpy as np
def rbjac(A, b, iguess,itr, w, grid):
    A = A.todense()
    x = iguess
    r = eject(grid)
    rb = np.ones_like(b) - r
    for it_count in range(itr):
        x_new = np.zeros_like(x)
        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        x = r * (x + w*(x_new - x)) + rb * x
        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        x = x + w*(x_new - x)
    return x

def eject(grid):
    n = (grid[0] - 1) * (grid[1] - 1) * (grid[2] - 1)
    b_new = np.zeros((n , 1))
    for i in range(0, n, 2):
        b_new[i] = 1
    return b_new