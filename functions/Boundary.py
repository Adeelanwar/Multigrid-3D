import scipy as sp
import numpy as np
from Project3D import *
def interiorpoints(grid, domain):
    h_x = (domain[0][1] - domain[0][0]) / grid[0]
    h_y = (domain[1][1] - domain[1][0]) / grid[1]
    h_z = (domain[2][1] - domain[2][0]) / grid[2]
    n_x = sp.ones((grid[0] - 1, 1))
    n_y = sp.ones((grid[1] - 1, 1))
    n_z = sp.ones((grid[2] - 1, 1))
    x_cor = np.array([np.arange(domain[0][0] + h_x, domain[0][1], h_x)]).T
    y_cor = np.array([np.arange(domain[1][0] + h_y, domain[1][1], h_y)]).T
    z_cor = np.array([np.arange(domain[2][0] + h_z, domain[2][1], h_z)]).T
    _x = np.kron(np.kron(n_y, n_z), x_cor)
    _y = np.kron(n_z, np.kron(y_cor, n_x))
    _z = np.kron(z_cor, np.kron(n_x, n_y))
    return _x, _y, _z
def interiorpoints2D(grid, domain):
    h_x = (domain[0][1] - domain[0][0]) / grid[0]
    h_y = (domain[1][1] - domain[1][0]) / grid[1]
    n_x = sp.ones((grid[0] - 1, 1))
    n_y = sp.ones((grid[1] - 1, 1))
    x_cor = np.array([np.arange(domain[0][0] + h_x, domain[0][1], h_x)]).T
    y_cor = np.array([np.arange(domain[1][0] + h_y, domain[1][1], h_y)]).T
    _x = np.kron(n_y, x_cor)
    _y = np.kron(y_cor, n_x)
    return _x, _y
def source(x, y, z):
    c1 = 3 * np.pi
    c2 = 3 * np.pi
    c3 = 3 * np.pi
    c4 = 3 * np.pi
    f1 = np.true_divide(((c1 * np.pi)**2 * np.sin(c1*np.pi*x) + (c2 * np.pi)**2 * np.sin(c2*np.pi*y) + (c3 * np.pi)**2 * np.sin(c3*np.pi*z))  ,(c4 + x + y + z))
    f2 = np.true_divide(2 * ((c1 * np.pi) * np.cos(c1 * np.pi * x) + (c2 * np.pi) * np.cos(c2 * np.pi * y) + (c3 * np.pi) * np.cos(c3 * np.pi * z) - 4 * analytic(x, y, z)) , np.square(c4 + x + y + z))
    return f1 + f2
def analytic(x, y, z):
    c1 = 3 * np.pi
    c2 = 3 * np.pi
    c3 = 3 * np.pi
    c4 = 3 * np.pi
    u = np.true_divide((np.sin(c1*np.pi*x) + np.sin(c2*np.pi*y) + np.sin(c3*np.pi*z))  , (c4 + x + y + z))
    return u

#def analytic(x, y, z):
#    return 1
#def source(x, y, z):
#    return 0

def boundary_contribution2D(grid,domain,n,z):
    h_x = (domain[0][1] - domain[0][0]) / grid[0]
    h_y = (domain[1][1] - domain[1][0]) / grid[1]
    nx = grid[0] - 1
    ny = grid[1] - 1
    e = np.ones((nx, 1))
    e2 = np.array([1])
    e1 = np.array([np.concatenate([e2, np.zeros(nx - 1)])]).T
    e2 = np.array([np.concatenate([np.zeros(ny - 1), e2])]).T
    x_cor = np.array([np.arange(domain[0][0] + h_x, domain[0][1], h_x)]).T
    y_cor = np.array([np.arange(domain[1][0] + h_y, domain[1][1], h_y)]).T
    abscissa = np.kron(e,x_cor)
    ordinates = np.kron(y_cor,e)
    if n == 1:
        r = np.kron(e1,e)
        x = r * abscissa
        y = domain[1][0] * r
    elif n== 2:
        r = np.kron(e,e2)
        y = r * ordinates
        x = domain[0][1] * r
        
    elif n==3:
        r = np.kron(e2,e)
        y = r * domain[1][1]
        x = abscissa * r
        
    elif n==4:
        r = np.kron(e,e1)
        x = r * domain[0][0]
        y = ordinates * r
    return r * analytic(x, y, z)

def boundary(grid, domain):
    level = 1
    h_x = (domain[0][1] - domain[0][0]) / grid[0]
    h_y = (domain[1][1] - domain[1][0]) / grid[1]
    h_z = (domain[2][1] - domain[2][0]) / grid[2]
    nx = grid[0] - 1
    ny = grid[1] - 1
    nz = grid[2] - 1
    x, y = interiorpoints2D(grid, domain)
    sol = np.zeros((nx * ny * nz , 1))
    bz = analytic(x, y, domain[2][0])
    b = np.zeros((nx * ny , 1))
    l = np.arange(domain[2][0] + h_z, domain[2][1], h_z)
    for z in l:
        bs = boundary_contribution2D(grid,domain,1,z)
        be = boundary_contribution2D(grid,domain,2,z)
        bn = boundary_contribution2D(grid,domain,3,z)
        bw = boundary_contribution2D(grid,domain,4,z)
        f = np.zeros((nz, 1))
        f[level - 1] = 1
        b = (1/(h_y**2))*(be + bn) + (1/(h_x**2))*(bs + bw)
        if level == 1:
             bz = analytic(x, y, domain[2][0])
             b += (1 / (h_z)**2) * bz
        if level == nz:
            bz = analytic(x, y, domain[2][1])
            b += (1 / (h_z)**2) * bz
        sol += np.kron(f, b)
        level += 1
    return sol
#testing prologation matrix and interpolation matrix.
def linear3d(x, y, z):
    return (x + 4) * (6*y + 2) * (3*z + 2)

##test
#gridspace2 = np.array([8, 8, 8])
#gridspace = np.array([16, 16, 16])
#domain = [(1, 2), (1, 2), (1, 2)]
#x1, y1, z1 = interiorpoints(gridspace2, domain)
#x2, y2, z2 = interiorpoints(gridspace, domain)
#u1 = linear3d(x1, y1, z1)
#u2 = linear3d(x2, y2, z2)
##
#P = prol3d(gridspace)
#print('here')
#R = rest3d(gridspace)
#print((R @ u2) - u1)
#h = (P @ u1) - u2
#print(h)
#P = P.todense()
