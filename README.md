# newton-sysid2015

This code was downloaded from < http://users.isy.liu.se/en/rt/manko/ > or < http://users.isy.liu.se/en/rt/johda87/ > and contains the code used to produce the results in the paper

* M. Kok, J. Dahlin, T. B. Schön and A. Wills, *Newton-based maximum likelihood estimation in nonlinear state space models*. Proceedings of the 17th IFAC Symposium on System Identification, Beijing, China, October 2015. 

The papers are available as a preprint from < http://arxiv.org/pdf/1502.03655v2 >, < http://users.isy.liu.se/en/rt/manko/ > and < http://users.isy.liu.se/en/rt/johda87/ >.

Requirements
--------------
For the linearization approximations, the program is written in Matlab 2014b. 

For the sampling approximations, the program is written in Python 2.7 and makes use of NumPy 1.7.1, SciPy 0.12.0, Matplotlib 1.2.1, Pandas. Please have these packages installed, on Ubuntu they can be installed using "sudo pip install --upgrade *package-name* ".

Included folders
--------------
**linearizationApproximations**
Contains code to recreate a subset of the numerical illustrations in the paper. Executes the proposed algorithm for Newton optimisation using the proposed linearization approximation. For more details, see the README.md file in this folder. 

**samplingApproximations**
Contains code to recreate a subset of the numerical illustrations in the paper. Executes the proposed algorithm for Newton optimisation using the proposed sampling approximations. For more details, see the README.md file in this folder. 
