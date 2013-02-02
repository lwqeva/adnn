%% adnn.git
%   Compute residual for 1-hidden layer nnet. Compatible with ADMAT.
function r = nn_residual(w,lExtra)
%   w           - weights of nnet
%   Extra.X     - sample input
%   Extra.Y     - sample output
%   Extra.dim   - number of nodes in hidden layer

global Extra;
if nargin > 1 && size(lExtra,1) > 1   % use local value if it's provided
    Extra = lExtra;
end
nx = size(Extra.X,1);  % nodes in input layer
nh = Extra.dim(1);       % nodes in the hidden layer

r = zeros(1,size(Extra.Y,2));  % residual

%W = vec2mat(w(1:(end-nh-1)),nx);  % make weights of hidden layer a matrix
for i = 1:nh
    W(i,1:nx) = w(((i-1)*nx+1):i*nx)';
end
a = w((end-nh):end)';          % weights of last layer


r = Extra.Y - (a(2:end) * sigmoid(W*Extra.X) + a(1) );

function x =sigmoid(x)
x = 1./(1+exp(-x));