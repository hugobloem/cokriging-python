import numpy as np

def cokri(x,xO,model,c,itype,avg,block,nd, ival,nk,rad,ntok):
    '''
    COKRI performs point or block cokriging in D dimensions (any integer) 
    of P variables (any integer) with a combination of R basic models 
    (any integer). 

    Syntax: 
    [x0s,s,sv,id,l]=cokri(x,x0,model,c,itype,avg,block,nd,ival,nk,rad,ntok) 

    Input description: 
        x:  The n x (p+d} data matrix. This data matrix can be imported from an 
            existing ascii file. Missing values are coded 'nan' (not-a-number). 
        x0: The m x d matrix of coordinates of points to estimate. 
        model: Each row of this matrix describes a different elementary structure. 
            The first column is a code for the model type, the d following 
            columns give the ranges along the different coordinates and the 
            subsequent columns give rotation angles ( a maximum of three}. 
            The codes for the current models are: 
                1: nugget effect 
                2: exponential model 
                3: gaussian model 
                4: spherical model 
                5: linear model 
            Note: a linear model is specified by arbitrary ranges and a sill 
            such that sill/range gives the desired slope in the direction 
            condidered. 
        c:  The (rp x p) coefficient matrix of the coregionalization model. 
            Position (i,j) in each submatrix of size p x p give the sill of the 
            elementary component for each cross-variogram (variogram} between 
            variable i and variable j. 
        itype: Code to indicate which type of cokriging is to be performed: 
                1: simple cokriging 
                2: ordinary cokriging with one nonbias condition 
                (Isaak8 and Srivastava). 
                3: ordinary cokriging with p nonbias condition. 
                4: universal cokriging with drift of order 1. 
                5: universal cokriging with drift of order 2. 
                99: cokriging i8 not performed, only sv is computed. 
        block: Vector (1 x d), giving the size of the block to estimate; 
            any values when point cokriging is required. 
        nd: Vector (1 x d), giving the discretization grid for block cokriging; 
            put every element equal to I for point cokriging. 
        ival: Code for cross-validation. 
                0: no cross-validation 
                1: cross-validation is performed by removing one variable at a 
                time at a given location. 
                2: cross-validation is performed by removing all variables at a 
                given location. 
        nk: Number of nearest neighbors in x matrix to use in the cokriging 
            (this includes locations with missing values even if all variables 
            are missing). 
        rad: Search radius for neighbors. 
        ntok: Points in x0 will be kriged by groups of ntok grid points. 
            When ntok>l, the search will find the nk nearest samples within 
            distance tad from the current ntok grid points centroid. 

    Output description: 

        For the usual application, only x0s and s are required and the other 
        output matrices may be omitted. 

        x0s: m x (d+p) matrix of the m points (blocks} to estimate by the 
            d coordinates and p cokriged estimates. 
        s:  m x (d+p} matrix of the m points (blocks} to estimate by the 
            d coordinates and the p cokriging variances. 
        sv: 1 x p vector of variances of points (blocks) in the universe. 
        id: (nk x p) x 2 matrix giving the identifiers of the lambda weights for 
            the last cokriging system solved. 
        b:  ((nk x p) + nc) x (ntok x p} matrix with lambda weights and 
            Lagrange multipliers of the last cokriging system solved. 
    '''
    
    # definition of some constants
    m, d = x0.shape

    # check for cross-validation
    if ival >= 1:
        ntok = 1
        x0 = x[:, :d]
        nd = np.ones((1,d))
        m, d = x0.shape
    rp, p = c.shape
    n, t = x.shape
    nk = np.minimum(nk, n)
    ntok = np.minimum(ntok, m)
    idp = np.arange(p).reshape((1,p))
    ng = np.prod(nd)

    # compute point (ng=1) or block (ng>1) variance
    for i in range(d):
        n1 = np.prod(nd[:i-1])
        nr = np.prod(nd[i+1:d])
        t = np.arange( .5*(1/nd[i]-1), .5*(1-1/nd[i]), 1/nd[i]).reshape((1,nd[i]))
        t2 = np.block([t2, np.kron(np.ones((n1, 1), np.kron(t, np.ones((nr, 1)) )))])
    
    grid = t2 * (np.ones((ng, 1)) @ block)
    t = np.block([grid, np.zeros((ng, p))])

    # for block cokriging a double grid is created by shifting slightly the
    # original grid to avoid the zero distance effect (Journel and Huijbregts, p. 96)
    if  ng > 1:
        grid += np.ones((ng, 1)) @ block /  (ng*1e6)
    x0s, s, id, l, k0 = cokri2(t, grid, [], model, c, sv, 99, avg, ng)

    # sv contain the variance of points or blocks in the universe
    for i in range(p):
        sv = np.block([sv, np.mean(k0[i:p*ng:p, i:p*ng:p])]) # TODO: check means
    
    # start cokriging
    for i in range(0, m, ntok):
        nnx = np.minimum(ntok, m-i)
        print(f'kriging points #{i+1} to #{i+nnx}')

        # sort x samples in increasing distance relatively to centroid of 'ntok'
        # points to krige
        centx0 = np.ones((n, 1)) @ np.mean(x0[i:i+nnx, :], axis=0) # TODO: check axis
        tx = (x[:, :d] - centx0) * (x[:, :d] - centx0) @ np.ones((d, 1))
        tx, j = np.sort(tx), np.argsort(tx)

        # keep samples inside searh radius; create an identifier of each sample
        # and variable (id)
        t = []
        id = []
        ii = 1
        tx = np.block([[tx], [np.nan]])
        while ii <= nk and tx[ii] <= rad*rad:
            t = np.block([[t], [x[j[ii], :]]])
            id = np.block([[id], [np.ones((p, 1)) @ j[ii], idp]])
            ii += 1
        t2 = x0[i:i+nnx, :]

        # if block cokriging discretize the block
        t2 = np.kron(t2, np.ones((ng, 1))) - np.kron(np.ones((nnx, 1)), grid)

        # check for cross-validation
        if ival >= 1:
            est = np.zeros((1, p))
            sest = np.zeros((1, p))

            # each variable is cokriged in its turn
            np = 1 if ival == 1 else p
            for ip in range(0, p, np):
                # because of the sort, the closest sample is the sample to
                # cross-validate and its value is in row 0 of t; a temporary vector
                # keeps the original values before performing cokriging.

                vtemp = t[0, d+ip:d+ip+np]
                t[0, d+ip:d+ip+np] = np.nan
                x0ss, ss, = cokri2(t, t2, id, model, c, sv, itype, avg, ng)
                est[ip:ip+np] = x0ss[ip:ip+np]
                sest[ip:ip+np] = ss[ip:ip+np]
                t[0, d+ip:d+ip+np] = vtemp
            x0s = np.block([[x0s], [t2,est]])
            s = np.block([[s], [t2,sest]])
        else:
            x0ss, ss, id, _ = cokri2(t, t2, id, model, c, sv, itype, avg, ng) # TODO: check _
            x0s = np.block([[x0s], [x0[i:i+nnx, :], x0ss]])
            s = np.block([[s], [x0[i:i+nnx, :], ss]])
    return x0s, s, sv, id, 1 # TODO: check 1
