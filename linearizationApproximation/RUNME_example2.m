function [theta,time,iters] = RUNME_example2(varargin)

addpath('tools') % Add the tools folder on the path

% Optionally turn of the numerical optimization since it is using Matlab's
% optimization toolbox
if nargin < 1
    NUM = 1;
else
    NUM = varargin{1};
end

% Optionally change the data data that is being used
if nargin < 2
    iDataSet = 1;
else
    iDataSet = varargin{2};
end

%% Load the data set 
% The first column contains the true states x, the second the measurements y
dataSet = ['../data/example2_T1000/data' num2str(iDataSet) '.txt'];
data = load(dataSet);
y = data(:,2)';

%% Parameter estimation
% model.A = 1; % Constant multiplication factor in front of the nonlinear dynamics
model.Q = 1; % Process noise covariance
model.hbias = 0;
model.R = 0.01; % Measurement noise covariance
model.x0 = 0; % Initial state
model.P0 = 1; % Initial state covariance
model.case = 2; % To select the second model

% Initialize the parameters theta
theta0(1,1) = 0.5;
theta0(2,1) = 0.7;

%% Compute the parameter estimates

if NUM
    % Use numerical gradients and the Matlab optimization toolbox to
    % estimate the parameters
    options = optimoptions('fminunc','Display','iter-detailed','Algorithm','quasi-Newton','HessUpdate','bfgs','OutputFcn',@myoutput);
    thetaIter_num = [];
    tic;
    [theta_num,~,~,output] = fminunc(@(theta) objectiveGradHess(model,y,theta),theta0,options);
    timeBFGSnum = toc;
    iterBFGSnum = output.iterations;
else
    theta_num = [];
    timeBFGSnum = [];
    iterBFGSnum = [];
    thetaIter_num = [];
end

% Use the estimated gradient and Hessian using linearization approximations
% as described in Section 3 of the paper. 
tic;
[theta_newton,nIter,thetaIter_newton] = newtonOpt(model,y,theta0);
timeNewton = toc;
iterNewton = nIter; 

% Save the results for output
theta = [theta_num',theta_newton'];
time = [timeBFGSnum timeNewton];
iters = [iterBFGSnum iterNewton];

%% Plot trace plot
figure(1), clf, 
plot(thetaIter_newton(:,1),'b*')
hold all
plot(thetaIter_newton(:,2),'b+')
plot(thetaIter_num(:,1),'r*')
plot(thetaIter_num(:,2),'r+')
xlabel('Iteration [#]')
ylabel('Parameter estimate \theta')
legend('ALG2 \theta_1','ALG2 \theta_2','NUM \theta_1','NUM \theta_2')

%% Helper function to save the estimates at the different iterations from the Matlab optimization toolbox
function stop = myoutput(theta_num,~,state)
    stop = false;
    if isequal(state,'iter')
      thetaIter_num = [thetaIter_num; theta_num'];
    end
end

end