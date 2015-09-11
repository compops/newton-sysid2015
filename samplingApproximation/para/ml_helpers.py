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

import numpy as np
import pandas
import os

##########################################################################
# Helper: compile the results and write to file
##########################################################################
def writeToFile_helper(ml,sm=None,fileOutName=None,noLLests=False):

    # Construct the columns labels
    if ( noLLests ):
        columnlabels = [None]*(ml.nPars+1);
    else:
        columnlabels = [None]*(ml.nPars+3);

    for ii in range(0,ml.nPars):
        columnlabels[ii]   = "th" + str(ii);

    columnlabels[ml.nPars] = "step";

    if ( noLLests == False ):
        columnlabels[ml.nPars+1] = "diffLogLikelihood";
        columnlabels[ml.nPars+2] = "logLikelihood";

    # Compile the results for output
    if ( noLLests ):
        out = np.hstack((ml.th,ml.step));
    else:
        out = np.hstack((ml.th,ml.step,ml.llDiff,ml.ll));

    # Write out the results to file
    fileOut = pandas.DataFrame(out,columns=columnlabels);
    fileOutName = 'results/' + str(ml.filePrefix) + '/' + str(ml.optMethod) + '_' + sm.filterType + '_' + sm.smootherType + '_N' + str(sm.nPart)  + '/' + str(ml.dataset) + '.csv';

    ensure_dir(fileOutName);
    fileOut.to_csv(fileOutName);

    print("writeToFile_helper: wrote results to file: " + fileOutName)

##############################################################################
# Check if dirs for outputs exists, otherwise create them
##############################################################################
def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

##############################################################################
##############################################################################
# End of file
##############################################################################
##############################################################################