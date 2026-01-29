function Trace_est = Stochastic_Lanczos_Quadrature(A, B, m)
%STOCHASTIC_LANCZOS_QUADRATURE(A,B,num_iterations) estimates the trace of
% matrix function. In this paper, the matrix function is the exponential
% matrix.
% 
%  Input:
%  A: symmetric matrix
%  B: matrix of initial vectors
%  m: Lanczos iteration steps
%
%  Output:
%  Trace_est: the trace estimation by the SLQ method
%
%  Usage (example):
%  A = gallery('wathen',10,10); B = rademacher(length(A),10); m = 20;
%  Trace_est = Stochastic_Lanczos_Quadrature(A,B,m);
%
%  Copyright: Shengxin Zhu, Beijing Normal University
%             Wenhao Li, Beijing Normal-Hongkong Baptist University

n = size(B,1);
d = size(B,2); 
beta = zeros(m,d);
alpha = zeros(m,d);
Trace_est = 0;

% Build Krylov subspace
K = zeros(n,d,m+1);
for k=1:d
    K(:,k,1) = B(:,k)/norm(B(:,k));
end

K(:,:,2) = A*K(:,:,1);
alpha(1,:) = sum(K(:,:,2).*K(:,:,1));
for i = 1:m-1
	K(:,:,i+1) = K(:,:,i+1) - bsxfun(@times,K(:,:,i),alpha(i,:));
	beta(i+1,:) = sqrt(sum(K(:,:,i+1).*K(:,:,i+1)));
	K(:,:,i+1) = bsxfun(@times,K(:,:,i+1),1./beta(i+1,:));
	K(:,:,i+2) = A*K(:,:,i+1) - bsxfun(@times,K(:,:,i),beta(i+1,:));
	alpha(i+1,:) = sum(K(:,:,i+2).*K(:,:,i+1));
end

for z=1:d
    T = diag(alpha(:,z)) + diag(beta(2:end,z),1) + diag(beta(2:end,z),-1);
	[V,D] = eig(T);
	Trace_est = Trace_est + V(1,:).^2 * exp(diag(D)) * norm(B(:,z))^2;
end
Trace_est = Trace_est/d;
