import numpy as np

def trans(cx, model, im):
    #
    # TRANS is called from COKRI2. It takes as input original coordinates and
    #       return the rotated and reduced coordinates following specifications
    #       described in the model[im, :]
    #

    # some constants are defined
    n, d = cx.shape
    m, p = model.shape

    # check for 1-D or isotropic model
    if p > d: # TODO: check if this is correct
        
        # perform rotation counterclockwise

        if  d==2:
            ang = model[im , 3]
            cang, sang = np.cos(ang/180*np.pi), np.sin(ang/180*np.pi)
            rot = np.array([[cang, -sang], [sang, cang]])
        else:
            # rotation matrix in 3D (the first three among d coordinates) is
            # computed around x, y and z in that order
            rot = np.eye(d)
            for i in range(3):
                ang = model[im, 4+i]
                cang, sang = np.cos(ang/180*np.pi), np.sin(ang/180*np.pi)
                rot2 = np.array([[cang, -sang], [sang, cang]])
                axe = np.ones((3, 1), dtype=bool)
                axe[i] = 0
                rot[axe, axe] = rot[axe, axe] @ rot2
        
        # rotation is performed around x, y and z in that order, the other coordinates
        # are left unchanged

        dm = np.minimum(3, d)
        cx[:, :dm] = cx[:, :dm] @ rot
        t = np.diag(np.block([model[im, 1:, dm+1], np.ones(d-dm, 1)]))
    else:
        t = np.eye(d) @ model[im, 1]
    
    # perform contractions or dilations (reduced h)
    t = np.maximum(t, 1e-10)
    cx = np.linalg.solve(cx, t)

    return cx

def means(x):
    '''
    MEANS Average or mean value. For column vectors MEANS(x) returns the mean
          value. For matrices or row vector, MEANS(x) is a row vector containing
          the mean value of each column. The only difference with MATLAB
          function mean is for a row vector, where MEANS returns the row vector
          instead of the mean value of the elements of the row.
    '''
    m, n = x.shape
    if m > 1:
        return np.mean(x, axis=0)
    else:
        return x

def checkmod(model, c, d, rad):
    '''
    CHECKMOD    This function generates 'ntot' points within a D-sphere of radius
                'rad' (in reduced rotated distances) and evaluates all
                cross-variograms and variograms as described in 'model' and 'c'.
                If the necessary condition |γij(h)| <= sqrt(γii(h) γjj(h)) is
                violated for any simulated point, then the message
                "Warning: the model is not admissible" appears.
    
    Input parameters: model: description of models (see COKRI)
                          c: matrices of sills (see COKRI)
                          d: dimension (1D, 2D, ...)
    '''

    # initialization
    t, p = c.shape
    r,t = model.shape
    ntot = 3000/np.maximum(d, p*p)

    # equations for the various variograms, models should be placed in the same
    # order as in COKRI2 note that here the variograms are computed instead 
    # of the covariograms as in COKRI2

    Gam = [
        lambda h: h!=0,                                                     # nugget
        lambda h: 1-np.exp(-h),                                             # exponential
        lambda h: 1-np.exp(-h**2),                                          # gaussian
        lambda h: 1.5 * np.minimum(h, 1)/1 - .5 * (np.minimum(h, 1)/1)**3,  # spherical
        lambda h: h                                                         # linear
    ]

    # generate 'ntot' random points inside a D-sphere of radius 'rad'
    t = np.random.uniform(-0.5*rad*2, 0.5*rad*2, (ntot, d))

    # evaluate variograms for each simulated point
    k = np.zeros((ntot*p , p))
    for i in range(r):
        t = trans(t, model, i)
        h = np.sqrt(np.sum(t**2, axis=1))
        ji, js = i*p, (i+1)*(p-1)
        g = Gam[model[i, 0]](h)
        k += np.kron(g, c[ji:js, :])

    # check that |γij| < sqrt(γii * γjj)
    for i in range(ntot):
        ii, it = i*p, (i+1)*p-1
        t = np.sqrt(np.diag(k[ii:it, :]))
        k[ii:it, :] = np.abs(k[ii:it, :]) - t @ t.T
    k = k > 1e-10
    if k.sum(k) > 0:
        print('Warning: the model is not admissible')
        return 0
    else:
        return 1