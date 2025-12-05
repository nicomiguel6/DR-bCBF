clear all; close all; clc;

syms x xdot theta thetadot % states
syms F_max u % control inputs
syms A B Q R % linearizations for LQR

g = 9.81
l = 0.5
mp = 0.1
mc = 1.0



% dynamics
xs = [x; xdot; theta; thetadot]

mt = mp + mc

den = l*(4.0/3.0 - 0.5*mp*cos(theta).^2/mt)

thetaddot_f = (g*sin(theta) - mp*l*thetadot.^2*sin(theta)*cos(theta))/(l*(4.0/3.0 - mp*cos(theta).^2/mt))

f_x = [xdot; mp*l*thetadot.^2*sin(theta)/mt - mp*l*cos(theta)*thetaddot_f/mt; thetadot; thetaddot_f];

g_x = [0.0; (1/mt) - (mp*l/mt)*cos(theta)*(-(cos(theta)/mt)/den); 0.0; -(cos(theta)/mt)/den];

xsdot = f_x + g_x*u

% Controls

tmp_A = g/(l*(4.0/3.0 - mp/mt))
A = [0, 1, 0, 0;
    0, 0, tmp_A, 0;
    0, 0, 0, 1;
    0, 0, tmp_A, 0];

tmp_B = -1/(l*(4.0/3.0 - mp/mt))
B = [0; 1/mt; 0; tmp_B];

% Define the cost matrices for LQR
Q = diag([1, 10, 1, 1]); % State cost
R = 1; % Control cost

[K, S, P] = lqr(A,B,Q,R)

% symbolic feedback control (closed loop)

fcl = f_x - g_x*K*xs

% log norm (calculate closed loop jacobian)

J_cl = jacobian(fcl, xs)

% mu_cl = eig(0.5*(J_cl + J_cl'))

