const MAX_ITER = 5000
const STOP_DIFF = 0.0001;

# a update(Newton Raphson)


function gradient(a,a_0,u,L,rho,b,y)
    x = length(y)
#	grad = grad+ (y[i]-invLogit(a+a_0))+(u' * L)[0:t_0] + rho*(L*a-append!(b, zeros(t-t_o,1)))
    grad = y.-invLogit(a+a_0)+(u' * L)' + rho*(L*a-b)   #transposing gradient
	return grad
end;



function hessian(a,a_0,rho,L)
	hess = Diagonal(vec((invLogit(a+a_0).*(1-invLogit(a+a_0)))))+rho*L
    return -1*hess
end;

invLogit(x) = 1./(1.+e.^-x)


function newton(y_i,a_0,L,rho,b)
    a = zeros(length(y_i),1)
    a_old = a
    iters = 0
    diff = 1.0
    while(diff >STOP_DIFF && iters< MAX_ITER )
        grad = gradient(a_old,a_0,u,L,rho,b,y_i)
        hess = hessian(a_old,a_0, rho,L)
        a = a_old - pinv(hess)*grad
        diff = norm(a-a_old)
        a_old = a
        iters = iters+1
    end
    return a
en

# b update(Soft Treshold)


function soft(a,u,rho, lambda,t,t_0)
    c= zeros(t_0)
    for i in (t_0+1):(t-1)
        c+ u[1:t_0]+rho*(L*a)[1:t_0]
    end
    c = c/(t-t_0)*2/rho
    return sign(c).*max(abs(c)-lambda/2,0)
end

#

function ADMM(A,g,t,t_0)
	L = laplacian_matrix(g)
	n= nv(g)
	a = zeros(t)
	b = zeros(t)
	u = zeros(t)
#	alpha = 1.5  #relaxation parameter
	iters = 0
	diff = 1.0
	a_old = a
	while(diff >STOP_DIFF && iters< MAX_ITER )
		#a update
		for y_i in A
			a = a + newton(y_i,a_0, L, rho, b)
		end

		#b update
		b_old = b
		b = soft(a,b,u,rho,lambda,t,t_0)


		#u update
		u = u+ rho*(L*a-b)
#		a_hat = alpha*a+(1-alpha)*b_old
#		u = u+ (a_hat-b)
		diff  = norm(a-a_old)
		a_old = a
	end
	return a
end