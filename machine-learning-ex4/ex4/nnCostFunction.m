function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1)); % 25x401 --- per L2 unit X bias plus one per L1 unit

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1)); % 10x26  --- per L3 unit X bias plus one per L2 unit

% Setup some useful variables
m = size(X, 1);


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

y = y == ones(m, num_labels).*1:1:num_labels; % 5000x10 --- per training set X one per num labels 

X = [ones(m, 1) X]; % 5000x401 --- per training set X bias plus one per attr
Z2 = X*Theta1'; % 5000x25 --- per training set X one per L2 unit --- weighted sums of X
A2 = sigmoid(Z2); % 500x  x 0x25 --- per training set X one per L2 unit  --- sigmoid val 0 - 1
A2 = [ones(m, 1) A2]; % 5000x26 --- per training set X bias plus one per L2 unit --- sigmoid val 0 - 1
Z3 = A2*Theta2'; % 5000x10 --- per training set X one per L3 unit --- weighted sums of A2
A3 = sigmoid(Z3); % 5000x10 --- per training set X one per L3 unit --- sigmoid val 0 - 1
Costs = -y.*log(A3)-(1-y).*(log(1-A3)); % 5000x10 --- per training set X one per num labels  --- diff between calc and actiual
Cost = sum(sum(Costs))/m; % 1x1 --- sum of diff between calc and actiual / 5000

Theta1NoBias = Theta1(:,2:end); % 25x400 --- per L2 unit X one per L1 unit--- no bias
Theta2NoBias = Theta2(:,2:end); % 10x25 --- per L3 unit X one per L2 unit--- no bias
RegTerm = lambda/(2*m)*(sum(sum(Theta1NoBias.*Theta1NoBias)) + sum(sum(Theta2NoBias.*Theta2NoBias))); % 1x1 reg term

J =  Cost + RegTerm;

Delta3 = A3 - y; % 5000x10 --- per training set X one per L3 unit --- delta actual to calculated
Delta2 = (Delta3 * Theta2)(:,2:end) .* sigmoidGradient(Z2);  % 5000x25 --- per training set X one per L2 unit --- delta actual to calculated

Delta_Theta2 = (A2' * Delta3)';  % 10x26 --- one per L3 unit X per training set --- delta actual to calculated
Delta_Theta1 = (X' * Delta2)'; % 25x401 --- one per L3 unit X per training set --- delta actual to calculated

Theta1_grad = Delta_Theta1/m + lambda*[zeros(size(Theta1,1), 1), Theta1NoBias]/m; % % 25x401 --- one per L3 unit X per training set - regularized
Theta2_grad = Delta_Theta2/m + lambda*[zeros(size(Theta2,1), 1), Theta2NoBias]/m; % 10x26 --- one per L3 unit X per training set - regularized

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
