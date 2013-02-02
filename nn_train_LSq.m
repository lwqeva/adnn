%% adnn.git
%   Train 1-hidden layer nnet with ADMAT lsqnonlin
function [w, Extra] = nn_train_LSq(X,Y,M)
%   X       - sample input
%   Y       - sample output
%   M       - number of nodes in the hidden layer
%   w       - the trained weight
%   Extra   - Extra contains information about the model


nx = size(X,1);     % number of nodes in input layer
N = size(X,2);    % num of samples


% training data and model info are passed via global variable
clear global Extra;
global Extra;
Extra.X = [ones(1,N);X]; % extend input for bias node
Extra.Y = Y;
Extra.dim = M;


w0 =  0.5*(rand(size(Extra.X,1)*M+M+1,1)-0.5);

options = optimset('lsqnonlin');
options = optimset(options,'Jacobian','on');

myfun = ADfun('nn_residual',length(w0));
[w, Res] = lsqnonlin(myfun, w0, [], [], options);