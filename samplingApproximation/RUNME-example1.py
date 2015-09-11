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

import numpy            as np
import matplotlib.pylab as plt
import time

from   state   import smc
from   para    import ml_opt
from   models  import newton_sysid2015_example1


##############################################################################
# Arrange the data structures
##############################################################################
sm               = smc.smcSampler();
ml               = ml_opt.stMLopt();


##############################################################################
# Setup the model
##############################################################################
sys              = newton_sysid2015_example1.ssm()
sys.par          = np.zeros((sys.nPar,1))
sys.par[0]       = 0.5;
sys.par[1]       = 0.3;
sys.par[2]       = 0.1;
sys.par[3]       = 1.0;
sys.T            = 1000;
sys.xo           = 0.0;


##########################################################################
# Generate data
##########################################################################

# Use the first data set
iDataSet = 0;

sys.generateData(np.zeros(sys.T),'../data/example1_T1000/data'+str(iDataSet+1)+'.txt',"xy");


##########################################################################
# Setup the parameters
##########################################################################

th               = newton_sysid2015_example1.ssm()
th.nParInference = 2;
th.copyData(sys);


##########################################################################
# Setup the SMC and Newton algorithms
##########################################################################

# Use the bootstrap particle filter
sm.filter            = sm.bPF;

# Set the number of particles, the lag and the Hessian estimator
sm.nPart             = 2000;
sm.fixedLag          = 20;

# Set the number of particles, the fixed-lag, no. backward paths
# and settings for the fast version based on rejection sampling
sm.nPaths            = 200;
sm.nPathsLimit       = 20;
sm.rho               = 0.5;

# Set the initial parameter guess, the number of iterations
ml.initPar           = ( 0.7, 0.0 )
ml.verbose           = True;
ml.maxIter           = 100;

# Set adaptive step size from the beginning with k * ml.gamma**(-ml.alpha)
# where k is the iteration number
ml.adaptStepSize     = True;
ml.adaptStepSizeFrom = 0;
ml.gamma             = 1.0;
ml.alpha             = 1.0/2.0;
ml.dataset           = iDataSet + 1;


##########################################################################
# Newton-based optimisation using the particle fixed-lag smoother
##########################################################################

# Set seed for reproducability
np.random.seed( 87655678 + iDataSet );

# Use the fixed-lag smoother
sm.smoother          = sm.flPS;

# Run the Newtown based optimisation routine
t0 = time.clock()

ml.newton(sm,sys,th)

te = time.clock() - t0;
print( "Run with FL smoother took: " + str( te ) + " seconds.")

# Write the result to file
ml.writeToFile(sm);

thFL   = np.array( ml.th, copy=True);


##########################################################################
# Newton-based optimisation using fast FFBSi
##########################################################################

# Set seed for reproducability
np.random.seed( 87655678 + iDataSet );

# Use the fast FFBSi smoother
sm.smoother          = sm.ffbsiPS;

# Run the Newtown based optimisation routine
t0 = time.clock()

ml.newton(sm,sys,th)

te = time.clock() - t0;
print( "Run with fast FFBSi smoother took: " + str( te ) + " seconds.")

# Write the result to file
ml.writeToFile(sm);

thFFBSi = np.array( ml.th, copy=True);


##########################################################################
# Plot the results
##########################################################################

plt.figure(1);

plt.subplot(2,1,1); plt.plot(thFL);
plt.title('fixed-lag particle smoother'); plt.xlabel('iteration'); plt.ylabel('parameter estimate');

plt.subplot(2,1,2); plt.plot(thFFBSi);
plt.title('fast FFBSi particle smoother'); plt.xlabel('iteration'); plt.ylabel('parameter estimate');


##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################
