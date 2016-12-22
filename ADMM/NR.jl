const MAX_ITER = 10000
const STOP_DIFF = 0.00001;

function gradient(beta,x,y)
    grad =zeros(size(x,2),1)
	for i in 1 : size(x,1)
		grad = grad+ x[i,:]*(y[i]-prob(x[i,:],beta))
	end
	return grad
end;


function hessian(beta, x, y)
	hess = zeros(size(x,2),size(x,2))
	for i in 1:size(x,1)
        hess=  hess + x[i,:]*x[i,:]'*maximum(prob(x[i,:],beta)*(1-prob(x[i,:],beta)))
	end
    return -1*hess
end;


function prob(x,beta)
	prob = exp(beta' * x)/(1+exp(beta' * x))
	return prob
end;