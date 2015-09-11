function [obj,grad,hess] = extendedKalmanSmoother_objective(model,y,x)
%% Computes the objective, gradient and Hessian of the smoothing problem
% Smoothing problem is solved as a nonlinear weighted least-squares problem

nStates = length(model.x0);
[nMeas,N] = size(y);

sP = chol(model.P0);
sQ = chol(model.Q);
sR = chol(model.R);    

e = zeros(nStates*N + nMeas*N,1); % Pre-allocate e's to be minimized
J = sparse(nStates*N + nMeas*N,nStates*N); % Pre-allocate Jacobians de/dtheta

C = model.C;
A = model.A; 

iE = 0;
for iSample = 1:N
    % Add residual and Jacobian for the initialization
    if iSample == 1
        iState = 1:nStates;
        iE = iE(end) + (1:nStates);
        J(iE,iState) = eye(nStates) / sP;
        e(iE) = (x(1:nStates) - model.x0) / sP ;
    end

    iState = (iSample-1)*nStates + 1 : (iSample-1)*nStates + nStates;
    iE = iE(end) + (1:nMeas);

    % Add residual and Jacobian for the measurement update
    dhdx = C;
    yhat = C*x(iState);
    if model.case == 1
        yhat = yhat + model.hbias;
    end

    e(iE) = ((y(:,iSample) - yhat)' / sR)';
    J(iE,iState) = -dhdx' / sR;

    % Add residual and Jacobian for the time update
    if iSample ~= N
        iE = iE(end) + (1:nStates);
        iStateNext = iSample*nStates + 1 : iSample*nStates + nStates;

        dgdx = 1/(x(iState)^2+1);
        xpred = A * atan(x(iState));

        e(iE) = ( x(iStateNext) - xpred ) / sQ;
        J(iE,iState) = - dgdx / sQ;
        J(iE,iStateNext) = eye(nStates) / sQ;
    end
end

% Combine e's can J's to form objective, gradient and Hessian of the
% nonlinear least squares objective function
obj = e'*e;
grad = J'*e;
hess = J'*J;

end