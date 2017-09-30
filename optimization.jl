module optimization

const MAX_ITER = 1000
const STOP_DIFF = 0.0001


soft(c,lambda) = sign(c).*max(abs(c)-lambda/2,0)

invLogit(x) = 1./(1.+e.^-x)   

function gradient(a,a_0,u,L,rho,b,y)
    grad = -1.*(y-invLogit(a+a_0))+L*u + rho*L*(L*a-b)
    return grad
end;

function hessian(a,a_0,rho,L)
    hess = Diagonal(vec((invLogit(a+a_0).*(1-invLogit(a+a_0)))))+rho*L^2
    return hess
end;


function newton(y,a_0,L,rho,b,u)
    a= zeros(length(y))
    a_old = a
    iters = 0
    diff = 1.0
    while(diff >STOP_DIFF && iters< MAX_ITER )
        grad = gradient(a_old,a_0,u,L,rho,b,y)
        hess = hessian(a_old,a_0, rho,L)
        a = a_old - inv(hess)*grad
        diff = norm(a-a_old)
        a_old = a
        iters = iters+1
    end
    if(iters == MAX_ITER)
        println("max iter reached")
    end
    return a
end


function gradient_descent(y,a_0,L,rho,b,u,step = 0.00001)
    a = zeros(length(y))
    a_old = a
    iters = 0
    diff = 1.0
    while(diff> STOP_DIFF && iters< MAX_ITER )
        grad = gradient(a_old,a_0,u,L,rho,b,y)
        a = a_old - grad*step
        diff = norm(a.-a_old)
        a_old = a
        iters = iters+1
    end
    if(iters == MAX_ITER)
        println("max iter reached")
    end
    return a
end


function gradient_descent_old(y,a_0,L,rho,b,u,step = 10)
    a = zeros(length(y))
    a_old = a
    iters = 0
    diff = 1.0
    grad = 1
    while(norm(grad)> STOP_DIFF && iters< MAX_ITER )
        grad = gradient(a_old,a_0,u,L,rho,b,y)
        a = a_old - grad*step
        a_old = a
        iters = iters+1
        println(norm(grad))
    end
    if(iters == MAX_ITER)
        println("max iter reached")
        return false
    end
    return a
end



function ADMM_grad(A,L, rho, lambda, a_0)
    t_0 = length(A[1])
    t = length(A[size(A)[1]])
    new = size(A)[1] 
    a = Array{Float64,1}[]
    u = Array{Float64,1}[]
    for i in t_0:t
        push!(a,zeros(i)+0.0)
        push!(u,zeros(i)+0.0)
    end
    b = zeros(t_0)
    iters = 0
    diff = 1.0
    b_old = b
    while(diff >STOP_DIFF && iters< MAX_ITER )
        for i in 1:new
            a[i] = gradient_descent(A[i],a_0,L[i],rho,vcat(b,zeros(i-1)),u[i],0.0001)
        end
        c = zeros(t)
        for i in 1:new 
            c[1:(t_0 +i-1)] = c[1:(t_0 +i-1)]+ (u[i]+rho*(L[i]*a[i]))/(rho*new)
        end
        b = soft(c[1:t_0],2*lambda/rho)
        #u update
        for i in 1:new
            u[i] = u[i]+ rho*(L[i]*a[i]-vcat(b,zeros(i-1)))
        end
        diff  = norm(b-b_old)
        b_old = b
        println(diff)
    end
    return(b)
end;
end