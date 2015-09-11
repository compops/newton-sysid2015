function mu = linesearch(objective,theta,p,model,y)
%% Line search algorithm to determine the step size. 
% Based on Nocedal & Wright, Numerical Optimization, 2006
mu = 1;
beta = 0.5;
alpha = 0.01;
[V_obj,grad] = objective(model,y,theta);
V_new = objective(model,y,theta+mu*p);
while V_new > (V_obj + alpha*mu*grad'*p) || ~isreal(V_new)
    mu = beta*mu; % Decrease step size
    V_new = objective(model,y,theta+mu*p);
    if mu <= 1E-16 % Make sure that we don't get stuck for too long in mu is approximately zero
        mu = 0;
        break;
    end
end    
end