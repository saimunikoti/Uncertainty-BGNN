function A2 = estimate_adjacency(n,M,PhiV, Phi_A, ASamp)

e = ones(n,1);
S = find(eye(n,n));

eta = 2.5;
lambda =100;

cvx_begin quiet
    variable A2(n,n) symmetric
  
    L_beta = diag(A2*e) - A2;
  
    minimize(trace(PhiV'*(L_beta)*PhiV)/M + eta*norm(A2(:),1) +  lambda*norm(Phi_A.*(A2) - ASamp ,'fro')) 
    subject to

     % constraints to ensure adjacency matrix is a feasible one \in \mathcal{A}
     L_beta*e == zeros(n,1);
     A2(:) >= 0;
     A2(S) == 0;
cvx_end

end
