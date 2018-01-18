module optimization_modified

const MAX_ITER = 300
const STOP_DIFF = 0.0001
const ADMM_MAX_ITER = 1000
const ADMM_STOP_DIFF = 1e-3

soft(c,lambda) = sign.(c).*max.(abs.(c)-lambda/2,0)

invLogit(x) = 1./(1.+e.^-x)   

function gradient(a,a_0,u,L,rho,b,y)
    grad = -1.*(y-invLogit(a.+a_0))+L*u + rho*L*(L*a-b)
    return grad
end;


function gradient_descent(a,y,a_0,L,rho,b,u,step = 0.0001)
    #a = zeros(length(y))
    a_old = a
    iters = 0
    diff = 1.0
    while(iters< MAX_ITER )
        grad = gradient(a_old,a_0,u,L,rho,b,y)
        a = a_old - grad*step
        diff = norm(a.-a_old)
        a_old = a
        iters = iters+1
    end
    #if(iters == MAX_ITER)
    #    println("max iter reached")
    #end    
    return a
end






function gradient_logreg(beta,x,y)
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
    prob = exp.((beta' * x)./(1+exp.(beta' * x)))[1]
    return prob
end;


function newton(x,y)
    b0 = mean(y)
    b = [[b0]' zeros(size(x,2)-1)']'
    bold = b
    iters = 0
    diff = 1.0
    while(diff >STOP_DIFF && iters< MAX_ITER )
        grad = gradient_logreg(bold,x,y)
        hess = hessian(bold,x,y)
        b = bold - pinv(hess)*grad
        diff = norm(b-bold)
        bold = b
        iters = iters+1
    end
    return b
end;



function ADMM_grad_para(A,L, rho, lambda, a_0,step)
    t_0 = length(A[1])
    t = length(A[size(A)[1]])
    new = size(A)[1] 
    a = SharedArray{Float64,2}(ones(t,new))
    u = Array{Float64,1}[]
    for i in t_0:t
        push!(u,zeros(i)+0.0)
    end
    b = zeros(t_0)
    iters = 0
    diff = 1.0
    b_old = b
    iters = 0
    a_0 = Array{Float64,1}[]
    for i in 1:new
        x = ones(size(A[i],1),2) 
        x[:,2] = diag(L[i])
        y = A[i]
        push!(a_0,vec(x*optimization_modified.newton(x,y)))
    end
    while(iters< ADMM_MAX_ITER )
        #if iters%10 ==0
        #    println(diff)
        #end
        a = @parallel hcat for i in 1:new
        temp = zeros(t)
        temp[1:t_0+i-1] = gradient_descent(a[1:t_0+i-1,i],A[i],a_0[i],L[i],rho,vcat(b,zeros(i-1)),u[i],step)
        temp
        end
            c = zeros(t)
        for i in 1:new 
            c[1:(t_0 +i-1)] = c[1:(t_0 +i-1)]+ (u[i]+rho*(L[i]*a[1:t_0+i-1,i]))/(rho*new)
        end
            b = soft(c[1:t_0],2*lambda/rho)
        #u update
        for i in 1:new
            u[i] = u[i]+ rho*(L[i]*a[1:t_0+i-1,i]-vcat(b,zeros(i-1)))
        end
        diff  = norm(b-b_old)
        println(diff)
        b_old = b
        iters += 1
    end
    return(b)
end;


end





    b0 = mean(y)
    b = [[b0]' zeros(size(x,2)-1)']'
    bold = b
    iters = 0
    diff = 1.0
    while(diff >STOP_DIFF && iters< MAX_ITER )
        print("sad")
        grad = optimization_modified.gradient_logreg(bold,x,y)
        print("aa")
        hess = optimization_modified.hessian(bold,x,y)
        b = bold - pinv(hess)*grad
        diff = norm(b-bold)
        bold = b
        iters = iters+1
    end


    grad =zeros(size(x,2),1)
    for i in 1 : size(x,1)
        grad = grad+ x[i,:]*(y[i]-optimization_modified.prob(x[i,:],beta))
    end

i = 100
x = ones(size(A[i],1),2) 
x[:,2] = diag(L[i])
y = A[i]
optimization_modified.newton(x,y)