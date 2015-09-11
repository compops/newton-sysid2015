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

import numpy       as np
import scipy.weave as weave

##########################################################################
# Resampling for SMC sampler: Systematic
##########################################################################

def resampleSystematic( w, N=0 ):
    code = \
    """ py::list ret;
	int jj = 0;
        for(int kk = 0; kk < N; kk++)
        {
            double uu  = ( u + kk ) / N;

            while( ww(jj) < uu && jj < H - 1)
            {
                jj++;
            }
            ret.append(jj);
        }
	return_val = ret;
    """
    H = len(w);
    if N==0:
        N = H;

    u   = float( np.random.uniform() );
    ww  = ( np.cumsum(w) / np.sum(w) ).astype(float);
    idx = weave.inline(code,['u','H','ww','N'], type_converters=weave.converters.blitz )
    return np.array( idx ).astype(int);

##########################################################################
# Resampling for SMC sampler: Multinomial
##########################################################################
def resampleMultinomial(w, N=0 ):
    code = \
    """ py::list ret;
	for(int kk = 0; kk < N; kk++)  // For each particle
        {
            int jj = 0;

            while( ww(jj) < u(kk) && jj < H - 1)
            {
                jj++;
            }
            ret.append(jj);
        }
	return_val = ret;
    """
    H = len(w);
    if N==0:
        N = H;

    u  = np.random.uniform(0.0,1.0,N);
    ww = ( np.cumsum(w) / np.sum(w) ).astype(float);
    idx = weave.inline(code,['u','H','ww','N'], type_converters=weave.converters.blitz)
    return np.array( idx ).astype(int);

##########################################################################
# Helper: Hessian estimators (Segal-Weinstein)
##########################################################################
def SegelWeinsteinInfoEstimator(sm,score,sys):
    s = np.sum(score, axis=1);
    infom = np.dot(np.mat(score),np.mat(score).transpose()) - np.dot(np.mat(s).transpose(),np.mat(s)) / sys.T;
    return infom;

##########################################################################
# Helper: calculate the gradient of the log-target
##########################################################################
def calcGradient(sm,sys,term1):

    # Check dimensions of the input
    if ( len(term1.shape) == 2):
        # Sum up the contributions from each time step
        gradient = np.nansum(term1,axis=1);
    else:
        gradient = term1;

    # Add the gradient of the log-prior
    for nn in range(0,sys.nParInference):
        gradient[nn]     = sys.dprior1(nn) + gradient[nn];

    # Write output
    sm.gradient  = gradient;
    sm.gradient0 = term1;

##########################################################################
# Helper: calculate the Hessian of the log-target
##########################################################################
def calcHessian(sm,sys,term1,term2=0,term3=0):

    sm.hessian = SegelWeinsteinInfoEstimator(sm,term1,sys)

    # Add the Hessian of the log-prior
    for nn in range(0,sys.nParInference):
        for mm in range(0,sys.nParInference):
            sm.hessian[nn,mm] = sys.ddprior1(nn,mm) + sm.hessian[nn,mm];

    # Write output
    sm.gradient0 = term1
    sm.hessian0  = term2
    sm.hessian1  = term3

##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################