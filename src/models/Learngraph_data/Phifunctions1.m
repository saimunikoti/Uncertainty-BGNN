function [Phi_A] = Phifunctions1(M,n, cmr_a)
Phi_A=[];
R=rand(n,n);
R = tril(R) + triu(R', 1);

Phi_A=(R< cmr_a);
Phi_A=double(Phi_A);

end

