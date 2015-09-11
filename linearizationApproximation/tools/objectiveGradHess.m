function [V,dVdt,dV2dt2] = objectiveGradHess(model,y,theta)
%% Computes the objective value, gradient and Hessian using the linearization 
% approximations as described in Section 3 of the paper.

N = length(y);
if model.case == 1 % Model 1, section 5.1
    model.C = theta(1);
    model.hbias = theta(2);
end
if model.case == 2 % Model 2, section 5.2
    model.A = theta(1);
    model.C = theta(2);
end
    
%% Compute objective value using an extended Kalman filter 
[yhat,S] = extendedKalmanFilter(model,y);
epsilon = y-yhat;
V = sum(1/2*arrayfun(@(i) epsilon(:,i)'*(S(:,:,i)\epsilon(:,i)) + log(det(S(:,:,i))),1:N));
    
%% If required, compute approximate Jacobian and Hessian using a smoother
if nargout >= 2
    s = zeros(N,length(theta));

    % Estimate smoothed state estimates and covariance by solving a MAP 
    % optimization problem
    [x,P,~] = extendedKalmanSmoother(model,y); 
    x = x';
    A = model.A;
    C = model.C;
    hbias = model.hbias;
    Q = model.Q;
    R = model.R;

    % Compute gradient
    if model.case == 1
        dCdtheta = 1;
        s(:,1) = 1/2 * arrayfun(@(i) trace( (P(:,:,i) + ...
            x(:,i)*x(:,i)') *  ...
            (dCdtheta(:,:,1)'/R*C + C'/R*dCdtheta(:,:,1))),1:N) - ...
            arrayfun(@(i) (y(i)-hbias) / R * dCdtheta(:,:,1) * x(:,i),1:N);
        dVdt(1,1) = sum(s(:,1));

        s(:,2) = - (y - C*x - hbias)./R;
        dVdt(2,1) = sum(s(:,2));
    elseif model.case == 2
        s(2:end,1) = - ((x(:,2:end) - A*atan(x(1:end-1)))/Q) .* atan(x(1:end-1));
        dVdt(1,1) = sum(s(:,1));

        dCdtheta = 1;
        s(:,2) = 1/2 * arrayfun(@(i) trace( (P(:,:,i) + ...
            x(:,i)*x(:,i)') *  ...
            (dCdtheta(:,:,1)'/R*C + C'/R*dCdtheta(:,:,1))),1:N) - ...
            arrayfun(@(i) y(i) / R * dCdtheta(:,:,1) * x(:,i),1:N);
        dVdt(2,1) = sum(s(:,2));
    end
    
    % Compute Hessian
    ssT = arrayfun(@(i) s(i,:)' * s(i,:),1:N,'UniformOutput',false);
    dV2dt2 = sum(cat(3,ssT{:}),3) - ...
        1/N * dVdt' * dVdt;
end

end