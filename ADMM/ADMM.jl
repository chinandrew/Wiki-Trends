const MAX_ITER = 5000
const STOP_DIFF = 0.0001;

# a update(Newton Raphson)

function gradient(a,a_0,u,L,rho,b)
    grad =zeros(size(x,2),1)
	for i in 1 : size(x,1)
		grad = grad+ (y[i]-invLogit(a+a_0))+u[0:t_0]' * L + rho*(L*a-append!(b, zeros(t-t_o,1)))
	end
	return grad
end;


function hessian(a,a_0,rho,L)
	hess = Diagonal(invLogit(a+a_0).*(1-invLogit(a+a_0))) + rho*L
    return -1*hess
end;

invLogit(x) = 1./(1.+e.^-x)


function newton(a,a_0,L,rho,b)
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

# b update(Soft Treshold)


function soft(a,b,u,rho, lambda)
	c= 0
	for i in (t_0+1):t
		c+ u[0:t_0]' * b+rho*b'*(L*a)[0:t_0]
	end
	c = c/(t-t_0)*2/rho
	return sign(c).*max(abs(c)-lambda/2,0)
end


#

function ADMM(A,g)
	L = laplacian_matrix(g)
	n= nv(g)
	a = zeros(n)
	b = zeros(n)
	u = zeros(n)
	alpha = 1.5  #relaxation parameter
	iters = 0
	diff = 1.0
	while(diff >STOP_DIFF && iters< MAX_ITER )
		#a update
		a_old = a
		for a_i in A
			a = a + newton(a_0, L, rho, b)
		end
		#b update
		b_old = b
		b = soft(a,b,u,rho,lambda,lambda)
		a_hat = alpha*a+(1-alpha)*b_old
		#u update
		u = u+ (a_hat-b)
	end
	return a
end