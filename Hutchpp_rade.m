function approx_logdet = Hutchpp_rade(A,num_iterations,num_queries)
dim = size(A,1);
Function = @(B) lanczosfAB(A, B, num_iterations);


sketch_frac = 1/3;
S_num_queries = round(num_queries * sketch_frac);
G_num_queries = round(num_queries * sketch_frac);


S = rademacher(dim, S_num_queries);


[Q,~] = qr(Function(S), 0); 


G = rademacher(dim, G_num_queries);
G = G - Q*(Q' * G);


% Compute Hutch++ Estimate value
approx_logdet = trace(Q' * Function(Q)) + ...
     (Stochastic_Lanczos_Quadrature(A,G,num_iterations));
