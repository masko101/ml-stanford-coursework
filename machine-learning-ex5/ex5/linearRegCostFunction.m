function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = X * theta;
diff = h - y;
cost = (1/(2*m))*sum(power(diff,2));
thetaNoConstant = theta(2:size(theta,1),:);
reg = (lambda/(2*m))*sum(power(thetaNoConstant,2));
J = cost + reg;

% =========================================================================
gradreg = [ 0; ones(size(theta,1)-1,1) ] .* ((lambda/m)*theta);
grad = (1/m)*(diff' * X)' + gradreg;
end
