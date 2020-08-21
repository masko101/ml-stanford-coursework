function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

CPerm = [ 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30 ];
sigPerm = [ 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30 ];
%CPerm = [ 0.009, 0.02, 0.06, 0.09, 0.2, 0.6, 0.9, 1, 2 ];
%sigPerm = [ 0.06, 0.08, 0.1, 0.2, 0.3 ];
perms =  horzcat((zeros(size(sigPerm,2), size(CPerm,2)) + CPerm)(:), (zeros(size(sigPerm,2), size(CPerm,2)) + sigPerm')(:));

results = []
for perm = perms'
  perm
  model = svmTrain(X, y, perm(1), @(x1, x2) gaussianKernel(x1, x2, perm(2)));
  predictions = svmPredict(model, Xval);
  error = mean(double(predictions ~= yval))
  results = [ results; perm' error ];
  %visualizeBoundary(X, y, model);
  %fprintf('Program paused. Press enter to continue.\n');
  %pause;
endfor
rSorted = sortrows(results,3);
rSorted(1)
C = rSorted(1, 1);
sigma = rSorted(1, 2);
% =========================================================================

end
