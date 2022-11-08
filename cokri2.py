import numpy as np

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
    cx = np.stack([x[:, :d], x0])

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
        k = np.stack([np.stack([k, np.ones(n*p+1)]), np.stack([np.ones((1, n*p)), 0])])
        k0, nc = np.stack([k0, np.zeros((1, m*p))]), 1
    elif itype >= 3:
        # ordinary cokriging (Myers, Math. Geol, 1982)
        t = np.kron(np.ones((1, n)), np.eye(p))
        k = np.stack([np.stack([k, t.T]), np.stack([t, np.zeros((p, p))])])
        k0, nc = np.stack([k0, np.kron(np.ones((1,m)), np.eye(p))]), p
        
        if itype >= 4:
            # universal kriging; linear drift constraints
            t = np.kron(cx[:n, :], np.ones((p,1)))
            k = np.block([k, [t, np.zeros((p,d))], [t.T, np.zeros((nca, nc+nca))]])
            t = np.kron(cx[n:n+m, :].T, np.ones((1,p)))
            k0 = np.block([[k0], [t]])
            nc += d
        
        if itype == 5:
            # universal kriging; quadratic drift constraints
            