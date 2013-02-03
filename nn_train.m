%% adnn
% nn_train: train a multilayer nnet. Current implementation only allow
% single output node.
function [nn, res] = nn_train(dim,X,Y)
% dim       - specification of dimensions of the neural nets.
% dim(i)    - number of nodes in the i-th hidden layer.
% X         - sample input
% Y         - sample output
% nn.dim    - copy of input dim
% nn.w      - trained weights
% res       - residual of current nnet on the training data


[w, res] = solve_LSq(dim,X,Y);

nn.dim = dim;
nn.w = w;