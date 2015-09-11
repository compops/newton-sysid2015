function [theta,nIter,thetaIter] = newtonOpt(model,y,theta)
%% Does Newton optimization
% Outputs: The resulting estimate theta, the amount of iterations nIter
% needed for convergence and the theta-estimates at the different
% iterations
% Inputs: The model parameters in the struct model, the measurements y 
% and the initial estimate theta

thetaIter = theta';

%% Do Newton update
iMax = 40;
for iIteration = 1:iMax
    % Compute the approximated cost function, gradient and Hessian
    [V,dVdt,dV2dt2] = objectiveGradHess(model,y,theta);

    % Correct for non-positive definite Hessians approximation
    if any(eig(dV2dt2) < 0)
        eigmin = min(eig(dV2dt2));
        dV2dt2 = dV2dt2 - 2*eigmin*eye(2);
    end

    % Compute step direction
    p = -dV2dt2\dVdt;

    % Compute 
    mu = linesearch(@objectiveGradHess,theta,p,model,y);

    % Update theta
    theta = theta + mu * p;

    % Save the previous theta's in case a trace plot is desired
    thetaIter = [thetaIter ; theta'];

    %% Display iteration progress
    disp(['Iteration: ' num2str(iIteration) ' Objective: ' num2str(V) ' Mu: ' num2str(mu) ' Step size: ' num2str(norm(mu*p))])        

    %% Stopping criterion
    if abs(norm(dVdt)) < 1E-4 || norm(mu*p) < 1E-3
        nIter = iIteration;
        break;
    end
end

end