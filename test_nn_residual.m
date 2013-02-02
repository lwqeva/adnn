%% test case for nn_residual.m
% test residual by examine the 1-hidden node nueral net.

nx = 1;     % number of nodes in input layer
N = 50;    % num of samples
M = 1;      % number of nodes in hidden layer

X = (rand(nx,N)-0.5)*2*pi;   % sample input
Y = sin(X);     % sample output

% training data and model info are passed via global variable
clear global Extra;
global Extra;
Extra.X = [ones(1,N);X]; % extend input for bias node
Extra.Y = Y;
Extra.dim = M;

%w =  0.5*(rand(size(Extra.X,1)*M+M+1,1)-0.5); %ones(size(Extra.X,1)*M+M+1,1);
w = [0;5;-0.5;1];
r = nn_residual(w);

subplot(3,1,1)
scatter(X,Y), axis([-4, 4, -2, 2]);
subplot(3,1,2)
scatter(X,Y-r), axis([-4, 4, -2, 2]);
subplot(3,1,3)
scatter(X,r), axis([-4, 4, -2, 2]);