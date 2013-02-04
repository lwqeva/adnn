% formulate the training as LSq problem and solve it.
function [w, res] = solve_LSq(dim,X,Y)

% training data and model info are passed via global variable Extra
clear global Extra;
global Extra;
Extra.dim = dim;
Extra.X = X;
Extra.Y = Y;

nx = size(X,1);  % nodes in input layer

% initial weights
w0 = rand( (nx+1)*dim(1)+dim(1)+1,1) - 0.5;

options = optimset('lsqnonlin');
options = optimset(options,'Jacobian','on');

myfun = ADfun('get_residual',length(w0));
[w, res] = lsqnonlin(myfun, w0, [], [], options);
clear global Extra;