function [yhat,S,xcorr] = extendedKalmanFilter(model,y)
%% Runs an EKF and outputs the predicted measurements, the residual 
% covariance and the state estimates

% Get all relevant parameters from the model struct
x0 = model.x0; % Initial state
P0 = model.P0; % Initial state covariance
C = model.C; % Multiplication in the measurement function
A = model.A; % Multiplication in front of the nonlinear dynamics
hbias = model.hbias; % Bias in the measurement function
Q = model.Q; % Process noise covariance
R = model.R; % Measurement noise covariance

N = length(y); % Length of the data set
yhat = zeros(1,N); % Predicted measurements
S = zeros(1,1,N); % Residual covariance
xpred = zeros(1,N); % Predicted x (x_{t | t-1})
xcorr = zeros(1,N); % Corrected x (x_{t | t})
Ppred = zeros(1,1,N); % Predicted P (P_{t | t-1})
Pcorr = zeros(1,1,N); % Corrected P (P_{t | t})
xpred(:,1) = x0; % Set the initial state
Ppred(:,:,1) = P0; % Set the initial covariance

%% Do EKF update
for i = 1:N   
    % Measurement update
    yhat(:,i) = C*xpred(:,i);
    if model.case == 1
        yhat(:,i) = yhat(:,i) + hbias;
    end

    S(:,:,i) = C * Ppred(:,:,i) * C' + R;
    K = Ppred(:,:,i) * C' / S(:,:,i);
    xcorr(:,i) = xpred(:,i) + K * (y(:,i) - yhat(:,i)); % Update state
    Pcorr(:,:,i) = Ppred(:,:,i) - K * S(:,:,i) * K'; % Update covariance

    % Time update
    dgdx = 1/(xcorr(:,i)^2+1);
    xpred(:,i+1) = A * atan(xcorr(:,i)); % Update state
    Ppred(:,:,i+1) = dgdx * Pcorr(:,:,i) * dgdx' + Q; % Update covariance
end

end