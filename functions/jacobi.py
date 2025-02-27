# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 17:08:51 2019

@author: Adeel
"""

import numpy as np

def jac(A, b, iguess,itr, w):
    A = A.todense()
    x = iguess
    for it_count in range(itr):
        x_new = np.zeros_like(x)
        for i in range(A.shape[0]):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        x = x + w*(x_new - x)
    return x