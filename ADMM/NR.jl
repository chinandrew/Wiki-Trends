

function gradient(a,a_0,u,L,rho,b,y)
#	grad = grad+ (y[i]-invLogit(a+a_0))+(u' * L)[0:t_0] + rho*L(L*a-append!(b, zeros(t-t_o,1)))
	grad = y.-invLogit(a+a_0)+(u' * L)' .+ rho*L*(L*a-b) 
	return grad
end;



function hessian(a,a_0,rho,L)
    hess = Diagonal(vec((invLogit(a+a_0).*(1-invLogit(a+a_0)))))+rho*L^2
    return -1*hess
end;

invLogit(x) = 1./(1.+e.^-x)


function newton(a_0,L,rho,b)
	a = zeros(nv(g),1)
    a_old = a
    iters = 0
    diff = 1.0
    while(diff >STOP_DIFF && iters< MAX_ITER )
        grad = gradient(a_old,a_0,L,rho,b)
        hess = hessian(a_old,a_0, rho,L)
        a = a_old - pinv(hess)*grad
        diff = norm(a-a_old)
        a_old = a
        iters = iters+1
    end
    return b
end