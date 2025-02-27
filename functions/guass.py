# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 15:40:08 2019

@author: Adeel
"""

import numpy as np
from scipy.sparse import triu, tril
import scipy.sparse as sp
def guass(A, b,iguess, itr):
    A = A.todense()
    x = iguess
    for it_count in range(itr):
        x_new = np.zeros_like(x)
        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        x = x_new
    return x

def Gauss_Seidel(A, b,iguess, itr):
    U = sp.triu(A, 1)
    L = sp.tril(A , 0)
    x = iguess
    for i in range(itr):
        xn = sp.linalg.inv(L) @ (b - U @ x)
        x = xn
    return x

