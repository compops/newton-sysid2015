# newton-sysid2015

This code was downloaded from < http://users.isy.liu.se/en/rt/manko/ > or < http://users.isy.liu.se/en/rt/johda87/ > and contains the code used to produce the results in the paper

* M. Kok, J. Dahlin, T. B. Schön and A. Wills, *Newton-based maximum likelihood estimation in nonlinear state space models*. Proceedings of the 17th IFAC Symposium on System Identification, Beijing, China, October 2015. 

The papers are available as a preprint from < http://arxiv.org/pdf/1502.03655v2 >, < http://users.isy.liu.se/en/rt/manko/ > and < http://users.isy.liu.se/en/rt/johda87/ >.

Requirements
--------------
The program is written in Matlab 2014b. The algorithm computes the parameter estimates using numerical gradients uses the Matlab Optimization toolbox. Note that it is possible to turn this algorithm off and only run the Newton optimization. 

Included files
--------------
**RUNME-example1.m**
Recreates a subset of the first numerical illustration in the paper. Executes the proposed algorithm for Newton optimisation with a linearization approximation where an EKF is used to compute the objective value and the smoothing problem is formulated as an optimization problem (Algorithm 2). Parameter estimates using a BFGS algorithm and numerical gradients (NUM) are also computed. The optimisation algorithm is run until convergence and the trace plot of the parameter estimates are displayed after the run. 

It is possible to only run algorithm 2 and not the BFGS algorithm (NUM, which uses the Matlab Optimization toolbox) by passing an input argument which is 0 for not running NUM (and 1 for running it). Note that the parameter inference is only made for dataset 1, whereas in the paper the same procedure is repeated for all the 100 data sets. A second input argument can be passed (between 1 and 100) to run a different data set. 

**RUNME-example2.m**
Recreates a subset of the second numerical illustration in the paper. All the details are the same as for example 1.

Supporting files
--------------
**tools/newtonOpt.m**
Main routine for Newton optimization.

**tools/objectiveGradHess.m**
Computes the objective value, gradient and Hessian using the linearization approximations as described in Section 3 of the paper for the two different state-space models.

**tools/extendedKalmanFilter.m**
Runs an EKF for the two different state space models.

**tools/extendedKalmanSmoother.m**
Main routine for the smoothing algorithm using linearization approximations.

**tools/extendedKalmaSmoother_objective.m**
Subroutine for the smoothing algorithm using linearization approximations. Compute the objective value, gradient and Hessian of the nonlinear weighted least-squares problem.

**tools/linesearch.m**
Subroutine for the optimization algorithms to determine the step size.