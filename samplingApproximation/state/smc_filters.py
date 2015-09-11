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
# Particle filtering: main routine
##########################################################################

def proto_pf(classSMC,sys):

    classSMC.T = sys.T;

    # Initalise variables
    a   = np.zeros((classSMC.nPart,sys.T));
    p   = np.zeros((classSMC.nPart,sys.T));
    pt  = np.zeros((classSMC.nPart,sys.T));
    v   = np.zeros((classSMC.nPart,sys.T));
    w   = np.zeros((classSMC.nPart,sys.T));
    xh  = np.zeros((sys.T,1));
    ll  = np.zeros(sys.T);

    # Generate initial state
    p[:,0] = sys.generateInitialState( classSMC.nPart );

    #=====================================================================
    # Run main loop
    #=====================================================================

    for tt in range(0, sys.T):
        if tt != 0:

            #=============================================================
            # Resample particles (systematic resampling)
            #=============================================================
            nIdx     = resampleSystematic(w[:,tt-1]);
            nIdx     = np.transpose(nIdx.astype(int));
            pt[:,tt] = p[nIdx,tt-1];
            a[:,tt]  = nIdx;


            #=============================================================
            # Propagate particles
            #=============================================================
            p[:,tt] = sys.generateState   ( pt[:,tt], tt-1);

        #=================================================================
        # Weight particles
        #=================================================================
        w[:,tt] = sys.evaluateObservation   ( p[:,tt], tt);

        # Rescale log-weights and recover weights
        wmax    = np.max( w[:,tt] );
        w[:,tt] = np.exp( w[:,tt] - wmax );

        # Estimate log-likelihood
        ll[tt]   = wmax + np.log(np.sum(w[:,tt])) - np.log(classSMC.nPart);
        w[:,tt] /= np.sum(w[:,tt]);

        # Estimate the filtered state
        xh[tt]  = np.sum( w[:,tt] * p[:,tt] );

    #=====================================================================
    # Create output
    #=====================================================================
    classSMC.xhatf = xh;
    classSMC.ll    = np.sum( ll );
    classSMC.llt   = ll;
    classSMC.w     = w;
    classSMC.v     = v;
    classSMC.a     = a;
    classSMC.p     = p;
    classSMC.pt    = pt;

##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################