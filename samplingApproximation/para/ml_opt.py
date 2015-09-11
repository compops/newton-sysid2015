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


import numpy      as     np
from   ml_helpers import *

##############################################################################
# Main class
##############################################################################

class stMLopt(object):

    ##########################################################################
    # Wrapper for Newton based optimisation
    ##########################################################################
    def newton(self,sm,sys,thSys):
        self.optMethod = "newton"
        self.direct_opt(sm,sys,thSys,True);

    ##########################################################################
    # Wrapper for write to file
    ##########################################################################
    def writeToFile(self,sm,fileOutName=None):
        writeToFile_helper(self,sm,fileOutName);

    ##########################################################################
    # Main routine for direct optimisation
    ##########################################################################
    def direct_opt(self,sm,sys,thSys,useHessian):

        #=====================================================================
        # Initalisation
        #=====================================================================

        # Set initial settings
        self.nPars      = thSys.nParInference;
        self.filePrefix = thSys.filePrefix;
        runNextIter     = True;
        self.iter       = 1;

        # Allocate vectors
        step        = np.zeros((self.maxIter,1))
        ll          = np.zeros((self.maxIter,1))
        llDiff      = np.zeros((self.maxIter,1))
        th          = np.zeros((self.maxIter,thSys.nParInference))
        thDiff      = np.zeros((self.maxIter,thSys.nParInference))
        gradient    = np.zeros((self.maxIter,thSys.nParInference))
        hessian     = np.zeros((self.maxIter,thSys.nParInference,thSys.nParInference))

        # Store the initial parameters
        thSys.storeParameters(self.initPar,sys);
        th[0,:]  = thSys.returnParameters();

        # Compute the initial gradient
        sm.smoother(thSys);
        ll[0]            = sm.ll;
        gradient[ 0, : ] = sm.gradient;
        hessian[0,:,:]   = sm.hessian;

        #=====================================================================
        # Main loop
        #=====================================================================
        while ( runNextIter  ):

            # Adapt step size
            step[self.iter,:] = self.gamma * self.iter**(-self.alpha);

            # Perform update
            th[self.iter,:] = th[self.iter-1,:] + step[self.iter,:] * np.dot( np.linalg.pinv( hessian[self.iter-1,:] ) , gradient[self.iter-1,:] );

            thDiff[self.iter] = th[self.iter] - th[self.iter-1,:];
            thSys.storeParameters(th[self.iter,:],sys);

            # Compute the gradient
            sm.smoother(thSys);
            ll[self.iter]          = sm.ll;
            gradient[self.iter,:]  = sm.gradient;
            hessian[self.iter,:,:] = sm.hessian;

            # Calculate the difference in log-likelihood and check exit condition
            llDiff[self.iter] = np.abs( ll[self.iter] - ll[self.iter-1] );

            # Update iteration number and check exit condition
            self.iter += 1;

            if ( self.iter == self.maxIter ):
                runNextIter = False;

            # Print output to console
            if (self.verbose ):
                parm = ["%.4f" % v for v in th[self.iter-1]];
                print("Iteration: " + str(self.iter) + " with current parameters: " + str(parm) + " and lldiff: " + str(llDiff[self.iter-1]) )

        #=====================================================================
        # Compile output
        #=====================================================================
        tmp         = range(0,self.iter-1);
        self.th     = th[tmp,:];
        self.step   = step[tmp,:]
        self.thDiff = thDiff[tmp,:]
        self.llDiff = llDiff[tmp,:];
        self.ll     = ll[tmp]

##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################
