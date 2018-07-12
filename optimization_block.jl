module optimization_block

const MAX_ITER = 150
const STOP_DIFF = 0.0001
const ADMM_MAX_ITER = 500
const ADMM_STOP_DIFF = 1e-3

soft(c,lambda) = sign.(c).*max.(abs.(c)-lambda/2,0)

invLogit(x) = 1./(1.+e.^-x)   


function gradient_descent(a,y,a_0,L,rho,b,u,step = 0.0001)
    a_old = a
    Lu = L*u
    rLL::SparseMatrixCSC{Float64,Int64}  = rho*L*L
    rLb::Array{Float64,1}  =rho* L*b
    #rLL  = rho*L*L
    #rLb  =rho* L*b
    for i in 1:MAX_ITER 
        grad = -1.*(y-invLogit(a_old.+a_0))+Lu + rLL*a_old - rLb
        a = a_old - grad*step
        a_old = a
    end
    return a
end



# can vectorize, but only runs once so havent bothered yet
function gradient_logreg(beta,x,y)
    grad =zeros(size(x,2),1)
    for i in 1 : size(x,1)
        grad = grad+ x[i,:]*(y[i]-prob(x[i,:],beta))[1]
    end
    return grad
end;

function hessian(beta, x, y)
    hess = zeros(size(x,2),size(x,2))
    for i in 1:size(x,1)
        hess=  hess + x[i,:]*x[i,:]'*maximum(prob(x[i,:],beta)[1]*(1-prob(x[i,:],beta)[1]))
    end
    return -1*hess
end;

function prob(x,beta)
    prob = exp.(beta' * x)./(1+exp.(beta' * x))
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
    num_blocks = 30
    a = SharedArray{Float64,2}(ones(t,num_blocks))
    u = Array{Float64,1}[]
    for i in t_0:t
        push!(u,zeros(i)+0.0)
    end
    b = zeros(t_0)
    iters = 0
    diff = 1.0
    b_old = b
    iters = 0
   # a_0 = Array{Float64,1}[]
   # for i in 1:new
   #     x = ones(size(A[i],1),2) 
   #     x[:,2] = diag(L[i])/sum(diag(L[i]))
   #     y = A[i]
   #     push!(a_0,vec(x*optimization_modified.newton(x,y)))
   # end
   # println("logreg complete")
    while(iters< ADMM_MAX_ITER )
        a = @parallel hcat for i in 1:num_blocks
            temp = zeros(t)
            end_index = floor(Int, t_0+(i) * (new/num_blocks)) -1
            index =floor(Int, (i)*(new/num_blocks))
            temp[1:end_index] = gradient_descent(a[1:end_index,i],A[index],a_0[index],L[index],rho,vcat(b,zeros(index-1)),u[index],step)
            temp
        end
        c = zeros(t)
        for i in 1:new 
            index = floor(Int, i/((new/num_blocks)+1))+1
            c[1:(t_0 +i-1)] = c[1:(t_0 +i-1)]+ (u[i]+rho*(L[i]*a[1:t_0+i-1,index]))/(rho*new)
        end
        b = soft(c[1:t_0],2*lambda/rho)
        #u update
        for i in 1:new
            index = floor(Int, i/((new/num_blocks)+1))+1
            u[i] = u[i]+ rho*(L[i]*a[1:t_0+i-1,index]-vcat(b,zeros(i-1)))
        end
        diff  = norm(b-b_old)
        println(diff)
        println(sum(b.>0))
        println(sum(b.!=0))
        b_old = b
        iters += 1
    end
    return(b)
end;


end



