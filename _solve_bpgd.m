% solve training problem by backpropagation
function [w, res] = solve_bpgd(dim,X,Y)

mu = 0.01;   % learning rate

nx = size(X,1)+1;  % nodes in input layer
nh = dim(1);       % nodes in the hidden layer

% initial weights
w = rand( (nx+1)*dim(1)+dim(1)+1,1) - 0.5;
% make weights of hidden layer a matrix
for i = 1:nh
    W(i,1:nx) = w(((i-1)*nx+1):i*nx)';
end
a = w((end-nh):end)';          % weights of last layer

X = [ones(1,size(X,2)); X];  % add bias node


[S, dS] = sigmoid(W*X);
R = Y - ( a(2:end) * S + a(1) ) ;
pr = R*R';

%% gradient descent via backpropagation
itmax = 400;
ratio = 1;
while ( (ratio > 1e-06) && (itmax > 0) )

    dW = zeros(nh,nx);
    for j = 1:nh
        for k = 1:size(X,1)
            dW(j,k) = R*(X(k,:).*dS(j,:))';
        end
    end
    W = W - mu*dW;
        
    a(1) = a(1) + mu*sum(R);
    a(2:end) = a(2:end) + mu*(R*S');

    [S, dS] = sigmoid(W*X);
    R = Y - ( a(2:end) * S + a(1) ) ;
    
    r = R*R';
    ratio = abs(r-pr)/r;
    itmax = itmax - 1;
end


% make weights of hidden layer a matrix
for i = 1:nh
    w(((i-1)*nx+1):i*nx) = W(i,1:nx)';
end
w((end-nh):end) = a';          % weights of last layer
res = r;

function [s, ds] = sigmoid(x)
s = 1./(1+exp(-x));
ds = s.*(1-s);