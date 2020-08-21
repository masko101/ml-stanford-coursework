function A_inv_b = matrixInverseVector(A, b, x_init, alpha)
  cost = []
  A_inv_b = x_init
  do 
    cost = (A*A_inv_b-b).^2
    A_inv_b = A_inv_b - alpha*cost
  until (norm(cost) < 10^-6)   
   
endfunction