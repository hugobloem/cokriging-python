import numpy as np
from utils import trans, means

def cokri2(x,x0,id,model,c,sv,itype,avg,ng):
    '''
    COKRI2: This function is celled from COKRI. The description for input and 
            output is given in COKRI. The only new variables are 'k0' which i8 
            the right member matrix of the cokriging system and 'ng' which is 
            the total number of points for block discretization. 
    '''

    # here we define the equations for the Various covariograms. Any new model 
    # can be added here.
    Gam = [
        lambda h: h==0,                                                 # nugget
        lambda h: np.exp(-h),                                           # exponential
        lambda h: np.exp(-h**2),                                        # gaussian
        lambda h: 1-(1.5*np.minimum(h,1)/1-.5*(np.minimum(h,1)/1)**3),  # spherical
        lambda h: 1-h                                                   # linear
    ]

    # definition of some constants
    n, t = x.shape
    rp, p = c.shape
    r = rp/p
    m, d = x0.shape
    cx = np.block([[x[:, :d]], [x0]])

    # calculation of left covariance matrix K and right covariance matrix K0
    K = np.zeros((n*p, (n+m)*p))
    for i in range(r):
        # calculation of matrix of reduced rotated distances H
        t = trans(cx, model, i)
        t = t @ t.T
        h = np.sqrt(-2*t + np.diag(t) @ np.ones((1, n+m)) + np.ones((n+m, 1)) @ np.diag(t).T)
        h = h[:n, :]
        ji, js = (i-1)*p+1, i*p # TODO: check indices

        # evaluation of the current basic structure
        g = Gam[model[i, 0]](h) # TODO: check if this is correct
        k += np.kron(g, c[ji:js, :])
    k0, k = k[:, n*p:(n+m)*p], k[:, :n*p]

    # constraints are added according to cokriging type
    if itype == 99:
        # no constraints
        pass
    if itype == 2:
        # cokriging with one non-bias condition (Isaaks and Srivastava, 1990)
        k = np.block([[k, np.ones(n*p+1)], [np.ones((1, n*p)), 0]])
        k0, nc = np.block([[k0], [np.zeros((1, m*p))]]), 1
    elif itype >= 3:
        # ordinary cokriging (Myers, Math. Geol, 1982)
        t = np.kron(np.ones((1, n)), np.eye(p))
        k = np.block([[k, t.T], [t, np.zeros((p, p))]])
        k0, nc = np.block([[k0], [np.kron(np.ones((1,m)), np.eye(p))]]), p
        
        if itype >= 4:
            # universal kriging; linear drift constraints
            t = np.kron(cx[:n, :], np.ones((p,1)))
            k = np.block([[k], [t, np.zeros((p,d))], [t.T, np.zeros((nca, nc+nca))]])
            t = np.kron(cx[n:n+m, :].T, np.ones((1,p)))
            k0 = np.block([[k0], [t]])
            nc += d
        
        if itype == 5:
            # universal kriging; quadratic drift constraints
            nca = d*(d+1)/2
            cx2 = []
            for i in range(d):
                for j in range(i, d):
                    cx2.append(cx[:, i]*cx[:, j])
            cx2 = np.array(cx2)
            t = np.kron(cx2[:n], np.ones((p, 1)))
            k = np.block([[k], [t, np.zeros((nc, nca))], [t.T, np.zeros((nca, nc+nca))]])
            t = np.kron(cx2[n:n+m, :].T, np.ones((1, p)))
            k0 = np.block([[k0], [t]])
            nc += nca

    # columns of k0 are summed up (if necessary) for block cokriging
    m = m/ng
    t = []
    for i in range(m):
        for ip in range(p):
            j = ng * p * (i-1) + ip # TODO: check indices
            t = np.block([t, means(k0[:, j:i*ng*p:p])])
    k0 = t
    t = x[:, d:d+p]
    if itype < 3:
        # if simple cokriging or cokriging with one non bias condition, the means
        # are substracted
        t = (t - np.ones((n, 1)) @ avg).T
    else:
        t = t.T
    
    # removal of lines and columns in k and k0 corresponding to missing values
    z = t.copy().reshape((n*p, 1))
    iz = ~np.isnan(z)
    iz2 = np.block([[iz], [np.ones(nc, 1)]])
    nz = np.sum(iz)

    # if no samples left return NaN
    if nz == 0:
        x0s = np.nan
        s = np.nan
        return x0s, s, id, b, k0
    else:
        k = k[iz2, iz2.T]
        k0 = k0[iz2, :]
        id = id[iz]
    
    # solution of the cokriging system by gauss elimination
    l = np.linalg.solve(k, k0)

    # calculation of cokrigin estimates
    t2 = l[:nz, :].T @ z[iz]
    t = t2.copy().reshape((p, m))

    # if simple or cokriging with one constraint, means are added back
    if itype < 3:
        t = t.T + np.ones((m, 1)) @ avg
    else:
        t = t.T
    x0s = t

    # calculation of cokriging variances
    s = np.kron(np.ones((m, 1)), sv)
    t = np.diag(l.T @ k0).reshape((p, m))
    s = s - t.T

    return x0s, s, id, b, k0