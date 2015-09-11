function [x,P,M] = extendedKalmanSmoother(model,y)
%% Extended Kalman smoother where the smoothing problem is solved as a MAP 
% optimization problem. Output: smoothed state x, smoothed state covariance
% P and smoothed state cross covariance M (P_{t,t-1 | N}. For details, see
% also the paper

% Run an EKF to get a fairly ok initial estimate for the smoother
[~,~,x] = extendedKalmanFilter(model,y);
x = x(:);

% Run the smoother as an optimization problem
nIter = 20;
for iIter = 1:nIter
    % Compute the objective, gradient and Hessian of the MAP problem
    [~,grad,hess] = extendedKalmanSmoother_objective(model,y,x);
    % Compute a step direction
    stepdir = -hess'\grad;  
    % Compute a step length
    mu = linesearch(@extendedKalmanSmoother_objective,x,stepdir,model,y);
    % Update the state
    x = x + mu * stepdir;
    % Stopping criterion
    if norm(mu*stepdir) < 1E-3
        break;
    end
end
    
% Compute P and M by taking the proper components from the inverse of the
% Hessian. Don't explicitly form the Hessian but instead compute components 
% column by column. 
N = length(y);
P = zeros(1,1,N);
M = zeros(1,1,N-1);
L = chol(hess,'lower'); % Cholesky factor can be precomputed
for iState = 1:N
    xi = zeros(N,1);
    xi(iState) = 1;
    H = L'\(L\xi);
    P(:,:,iState) = H(iState);
    if iState <= N-1
        M(:,:,iState) = H(iState + 1);
    end
end

end