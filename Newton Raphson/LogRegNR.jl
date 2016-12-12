MAX_ITER = 5000
STOP_DIFF = 0.0001

function prob(x,beta)
	prob = exp(beta' * x)/(1+exp(beta' * x))
	return prob
end

function gradient(beta,x,y)
	grad = 0
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
	n = size(x,1)
	b0 = mean(y)
	b = [b0 zeros(size(x,2)-1)']'
    bold = b
    iters = 0
    diff = 1
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
