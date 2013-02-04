clear;

nx = 1;     % number of nodes in input layer
N = 50;     % num of samples
M = 5;      % number of nodes in hidden layer

X = (rand(nx,N)-0.5)*2*pi;   % sample input
Y = sin(X)+cos(2*X);     % sample output

% train nnet
[nn, r] = nn_train(M,X,Y);

nn.X = X;
nn.Y = Y;

r = get_residual(nn.w,nn);

subplot(3,1,1)
scatter(X,Y), axis([-4, 4, -2, 2]);
subplot(3,1,2)
scatter(X,Y-r), axis([-4, 4, -2, 2]);
subplot(3,1,3)
scatter(X,r), axis([-4, 4, -2, 2]);
norm(r)