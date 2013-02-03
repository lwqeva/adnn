% compute the residual of nnet given weights and sample data
function r = get_residual(w,lExtra)
% r - the residual vector. Each entry is the residual for a sample pair.
% w - weights for each link in nnet.
% Extra.dim(1)  - number of nodes in the hidden layer.
% Extra.X       - sample input
% Extra.Y       - sample output

global Extra;
if nargin > 1 && size(lExtra,1) >= 1   % use local value if it's provided
    Extra = lExtra;
end
nx = size(Extra.X,1)+1;  % nodes in input layer
nh = Extra.dim(1);       % nodes in the hidden layer

% make weights of hidden layer a matrix
for i = 1:nh
    W(i,1:nx) = w(((i-1)*nx+1):i*nx)';
end
a = w((end-nh):end)';          % weights of last layer

X = [ones(1,size(Extra.X,2)); Extra.X];  % add bias node
r = Extra.Y - (a(2:end) * sigmoid(W*X) + a(1) );

function x =sigmoid(x)
x = 1./(1+exp(-x));