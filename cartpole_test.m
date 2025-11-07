clear all; close all; clc;

mc = 1
mp = 0.1
g = 10
l = 1

A = [0 0 1 0;
    0 0 0 1;
    0 mp*g/mc 0 0;
    0 g*(mc+mp)/(mc*l) 0 0 ];
B = [0 0 1/mc 1/(mc*l)]';

Q = diag([1, 1.1, 1, 1])

R = 1

K = lqr(A, B, Q, R);

% Define the state-space system
sys = ss(A, B, eye(4), 0);

% Simulate the response of the system to an initial condition
initialCondition = [0; 0.1; 0; 0]; % Initial state
t = 0:0.01:10; % Time vector
[y, t, x] = lsim(sys, ones(size(t)), t, initialCondition);

lqr_sys = ss(A-B*K, zeros(4,1), eye(4), 0);
% Simulate the response of the LQR controlled system
[y_lqr, t_lqr, x_lqr] = lsim(lqr_sys, ones(size(t)), t, initialCondition);

% Plot the system response
figure;
plot(t_lqr, y_lqr);
xlabel('Time (s)');
ylabel('System Response');
title('Response of the State-Space System');
grid on;