const MAX_ITER = 10000
const STOP_DIFF = 0.00001

function prob(x,beta)
	prob = exp(beta' * x)/(1+exp(beta' * x))
	return prob
end

function gradient(beta,x,y)
    grad =zeros(size(x,2),1)
    for i in 1 : size(x,1)
        grad = grad+ x[i,:]*(y[i]-prob(x[i,:],beta))
    end
    return grad
end

function hessian(beta, x, y)
    hess = zeros(size(x,2),size(x,2))
    for i in 1:size(x,1)
        hess=  hess + x[i,:]*x[i,:]'*maximum(prob(x[i,:],beta)*(1-prob(x[i,:],beta)))
    end
    return -1*hess
end




function newton(x,y)
    b0 = mean(y)
    b = [[b0]' zeros(size(x,2)-1)']'
    bold = b
    iters = 0
    diff = 1.0
    while(diff >STOP_DIFF && iters< MAX_ITER )
        grad = gradient(bold,x,y)
        hess = hessian(bold,x,y)
        b = bold - pinv(hess)*grad
        diff = norm(b-bold)
        bold = b
        iters = iters+1
    end
    return b
end

data =[ 17 1 ;  9 0; 24 1; 17 1; 15 1 ;11 1; 10 1; 15 0; 16 1 ;17 1; 13 1; 7 0; 14 1; 12 1; 10 1; 19 1; 13 1; 14 1; 8 1; 
 14 1 ;  8 1;  16 1; 23 1; 7 1; 8 0;12 0;13 1; 13 1; 12 1;  8 1 ;12 1 ;9 0; 10 1; 5 1; 6 0; 16 1; 10 1 ;12 0; 7 1; 14 0 ;
 10 0; 10 1; 8 1; 15 1;  8 0; 11 1; 11 1; 15 1; 16 1;  8 1; 9 0; 5 0;  3 1; 9 0; 8 1; 6 0; 5 1;12 1;11 0;5 0;8 0;14 1; 
  5 0 ;14 1; 13 1; 6 0;14 1; 5 0; 9 1; 7 1 ;20 1; 12 1; 7 0;12 0;11 1; 5 0 ; 3 0 ];

x = ones(size(data,1),2) 
x[:,2] = data[:,1];
y = data[:,2];

@code_warntype newton(x,y)