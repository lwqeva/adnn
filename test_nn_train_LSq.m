clear;

nx = 1;     % number of nodes in input layer
N = 50;    % num of samples
M = 2;      % number of nodes in hidden layer

X = (rand(nx,N)-0.5)*2*pi;   % sample input
Y = sin(X);     % sample output

% train nnet
[w, Extra] = nn_train_LSq(X,Y,M);

% compute residual
r = nn_residual(w,Extra);

subplot(3,1,1)
scatter(X,Y), axis([-4, 4, -2, 2]);
subplot(3,1,2)
scatter(X,Y-r), axis([-4, 4, -2, 2]);
subplot(3,1,3)
scatter(X,r), axis([-4, 4, -2, 2]);
norm(r)