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
from smc_filters             import *
from smc_smoothers           import *

##############################################################################
# Main class
##############################################################################

class smcSampler(object):

    ##########################################################################
    # Initalisation
    ##########################################################################

    # No particles in the filter and the number of backward paths in FFBSi
    nPart            = None;
    nPaths           = None;
    nPart2           = None;

    # For the rejection-sampling FFBSi with early stopping
    nPathsLimit      = None;
    rho              = None;

    # Lag for the fixed-lag smooother and Newey-West estimator for Hessian
    fixedLag         = None;

    ##########################################################################
    # Particle filtering: wrappers for special cases
    ##########################################################################

    def bPF(self,sys):
        self.filePrefix               = sys.filePrefix;
        self.resamplingInternal       = 1;
        self.filterTypeInternal       = "bootstrap"
        self.condFilterInternal       = 0;
        self.ancestorSamplingInternal = 0;
        self.filterType               = "bPF";
        self.pf(sys);

    ##########################################################################
    # Particle filtering and smoothing
    ##########################################################################

    # Auxiliuary particle filter
    pf           = proto_pf

    # Particle smoothers
    flPS         = proto_flPS
    ffbsiPS      = proto_ffbsiPS

##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################
