##############################################################################
##############################################################################
# Example code for Newton type maximum likelihood parameter inference
# in ninlinear state space models using particle methods.
#
# Please cite:
#
# M. Kok, J. Dahlin, T. B. Sch\"{o}n and A. Wills,
# "Newton-based maximum likelihood estimation in nonlinear state space models.
# Proceedings of the 17th IFAC Symposium on System Identification,
# Beijing, China, October 2015.
#
# (c) 2015 Johan Dahlin
# johan.dahlin (at) liu.se
#
# Distributed under the MIT license.
#
##############################################################################
##############################################################################
import numpy                 as     np
from smc_helpers             import *

##########################################################################
# Particle smoothing: fixed-lag smoother
##########################################################################
def proto_flPS(classSMC,sys):

    #=====================================================================
    # Initalisation
    #=====================================================================

    # Check algorithm settings and set to default if needed
    classSMC.T = sys.T;
    classSMC.smootherType = "fl"

    # Run initial filter
    classSMC.filter(sys);

    # Initalise variables
    xs    = np.zeros((sys.T,1));
    g1    = np.zeros((sys.nParInference,sys.T));

    #=====================================================================
    # Main loop
    #=====================================================================

    # Run the fixed-lag smoother for the rest
    for tt in range(0, sys.T-1):
        at  = np.arange(0,classSMC.nPart)
        kk  = np.min( (tt+classSMC.fixedLag, sys.T-1) )

        # Reconstruct particle trajectory
        for ii in range(kk,tt,-1):
            att = at.astype(int);
            at  = at.astype(int);
            at  = classSMC.a[at,ii];
            at  = at.astype(int);

        # Estimate state
        xs[tt] = np.sum( classSMC.p[at,tt] * classSMC.w[:, kk] );

        #=================================================================
        # Estimate gradient
        #=================================================================
        sa = sys.Dparm  ( classSMC.p[att,tt+1], classSMC.p[at,tt], np.zeros(classSMC.nPart), at, tt);

        for nn in range(0,sys.nParInference):
            g1[nn,tt]       = np.sum( sa[:,nn] * classSMC.w[:,kk] );

    #=====================================================================
    # Estimate additive functionals and write output
    #=====================================================================
    calcGradient(classSMC,sys,g1);
    calcHessian(classSMC,sys,g1);

    # Save the smoothed state estimate
    classSMC.xhats = xs;

##########################################################################
# Particle smoother: forward-filter backward-simulator (FFBSi)
##########################################################################
def proto_ffbsiPS(classSMC,sys,rejectionSampling=True,earlyStopping=True):

    #=====================================================================
    # Initalisation
    #=====================================================================

    # Check algorithm settings and set to default if needed
    classSMC.T = sys.T;
    classSMC.smootherType = "ffbsi"

    # Run initial filter
    classSMC.filter(sys);

    # Intialise variables
    v  = np.zeros((classSMC.nPart,sys.T));
    ws = np.zeros((classSMC.nPart,1));
    xs = np.zeros((sys.T,1));
    sa = np.zeros((classSMC.nPart,sys.nParInference));
    g1 = np.zeros((sys.nParInference,sys.T));

    # Initialise the particle paths and weights
    ps            = np.zeros((classSMC.nPaths,sys.T));
    nIdx          = resampleMultinomial(classSMC.w[:,sys.T-1])[0:classSMC.nPaths];
    ps[:,sys.T-1] = classSMC.p[nIdx,sys.T-1];

    xs[sys.T-1]   = np.mean( ps[:,sys.T-1] );

    #=====================================================================
    # Main loop
    #=====================================================================
    for tt in range(sys.T-2,0,-1):

        if ( rejectionSampling == False ):
            # Use standard formulation of FFBSi

            for jj in range(0,classSMC.nPaths):
                # Compute the normalisation term
                v[jj,tt] = np.sum( classSMC.w[:,tt] * sys.evaluateState( classSMC.p[jj,tt+1], classSMC.p[:,tt], tt) );

                # Compute the 1-step smoothing weights
                ws = classSMC.w[:,tt] * sys.evaluateState( ps[jj,tt+1], classSMC.p[:,tt], tt) / v[jj,tt];

                # Sample from the backward kernel
                pIdx = resampleMultinomial(ws)[0];

                # Append the new particle
                ps[jj,tt]  = classSMC.p[pIdx,tt];

        else:
            # Use rejection sampling
            L           = np.arange( classSMC.nPaths ).astype(int);

            if ( earlyStopping ):
                counterLimit = classSMC.nPathsLimit;
            else:
                counterLimit = classSMC.nPaths * 100;

            counterIter = 0;

            # As long as we have trajectories left to sample and have not reach the early stopping
            while ( ( len(L) > 0 ) & ( counterIter < counterLimit ) ):

                # Compute the length of L
                n = len(L);

                # Sample n weights and uniforms
                I = np.random.choice( classSMC.nPart, p=classSMC.w[:,tt], size=n  )
                U = np.random.uniform( size=n  )

                # Compute the acceptance probability and the decision
                prob   = sys.evaluateState( ps[L,tt+1], classSMC.p[I,tt], tt);
                accept = (U <= (prob / classSMC.rho) );

                # Append the new particle to the trajectory
                ps[ L[accept], tt ]  = classSMC.p[ I[accept], tt ];

                # Remove the accepted elements from the list
                L = np.delete( L, np.where( accept == True ) );

                counterIter += 1;

            # Print error message if we have reached the limit and do not have early stopping
            if ( ( earlyStopping == False) & ( counterIter == counterLimit ) ):
                raise NameError("To many iterations, aborting");


            # Use standard FFBSi for the remaing trajectories if we have early stopping
            if ( ( earlyStopping == True) & ( counterIter == counterLimit ) ):

                for jj in L:
                    # Compute the normalisation term
                    v[jj,tt] = np.sum( classSMC.w[:,tt] * sys.evaluateState( classSMC.p[jj,tt+1], classSMC.p[:,tt], tt) );

                    # Compute the 1-step smoothing weights
                    ws = classSMC.w[:,tt] * sys.evaluateState( ps[jj,tt+1], classSMC.p[:,tt], tt) / v[jj,tt];

                    # Sample from the backward kernel
                    pIdx = resampleMultinomial(ws)[0];

                    # Append the new particle
                    ps[jj,tt]  = classSMC.p[pIdx,tt];

        # Estimate state
        xs[tt] = np.mean( ps[:, tt] );

        # Gradient and Hessian of the complete log-likelihood
        sa  = sys.Dparm(  ps[:,tt+1], ps[:,tt], np.zeros(classSMC.nPaths), range(0,classSMC.nPaths), tt);

        g1[:,tt] = np.mean( sa, axis=0);

    #=====================================================================
    # Estimate additive functionals and write output
    #=====================================================================
    calcGradient(classSMC,sys,g1);
    calcHessian(classSMC,sys,g1);

    # Save the smoothed state estimate
    classSMC.xhats = xs;
    classSMC.ps    = ps;

##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################