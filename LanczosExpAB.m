function approximate_expAB= LanczosExpAB(A, B, iter)
% LANCZOSEXPAB(A, B, iter) approximates exp(A) * B by Lanczos iteration.
% 
% This is a modification of the Lanczos code available at
% 		https://github.com/RaphaelArkadyMeyerNYU/HutchPlusPlus/blob/
%       main/experiments/lanczos.m
%
% Required Inputs:
% - A: Square input matrix.
% 
% - B: Rectangular input matrix. A*B should be a valid computation.
% 
% - func: function_handle that given a real eigenvalue, returns a new eigenvalue.
%
% Examples:
% 
% Let A be a matrix with n rows and n columns.
% Let x be a column vector with n entries.
% Let B be a matrix with n rows and k columns.
% 
% Estimate e^A * x with 10 iterations of Lanczos:
% 	lanczos(A, x, @exp, 10)
% 
%  Copyright: Wenhao Li, Beijing Normal-Hongkong Baptist University

% Count dimensions. Allocate Space.
n = size(B,1);
d = size(B,2); 
beta = zeros(iter,d);
alpha = zeros(iter,d);

% Build Krylov subspace
K = zeros(n,d,iter+1);
for k=1:d
    K(:,k,1) = B(:,k)/norm(B(:,k));
end
K(:,:,2) = A*K(:,:,1);
alpha(1,:) = sum(K(:,:,2).*K(:,:,1));
for i = 1:iter-1
	K(:,:,i+1) = K(:,:,i+1) - bsxfun(@times,K(:,:,i),alpha(i,:));
	beta(i+1,:) = sqrt(sum(K(:,:,i+1).*K(:,:,i+1)));
	K(:,:,i+1) = bsxfun(@times,K(:,:,i+1),1./beta(i+1,:));
	K(:,:,i+2) = A*K(:,:,i+1) - bsxfun(@times,K(:,:,i),beta(i+1,:));
	alpha(i+1,:) = sum(K(:,:,i+2).*K(:,:,i+1));
end

% Approximate func(A)*B.
approximate_expAB = zeros(n,d);
for z=1:d
    T = diag(alpha(:,z)) + diag(beta(2:end,z),1) + diag(beta(2:end,z),-1);
	[V,D] = eig(T);
	fD = diag(exp(diag(D)));
    fAB_all = norm(B(:,z))*reshape(K(:,z,1:iter),n,iter)*V*fD*V';
	approximate_expAB(:,z) = real(fAB_all(:,1));
end
