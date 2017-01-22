using LightGraphs

invLogit(x) = 1./(1.+e.^-x)   

function addNode2(graph, p)
    add_vertex!(graph)
    x = nv(graph)
    degree = 0
    while degree ==0
        flips = rand(x-1)
        for i = 1:x-1
            if p[i]>flips[i]
                add_edge!(graph,i,x)
                degree +=1 
            end
        end
    end
    return graph
end

function addPrefNode(g,b,a_0 = -7)
    n = nv(g)
    L = laplacian_matrix(g)
    a = lufact(L) \ (b - mean(b))    
    p = invLogit(a+a_0)
    addNode2(g,p)
    push!(b,0)
    return g
end

function randEdgeGen(graph, newedges)
    for i in 1:newedges
        z = newedges
        x = collect(1:nv(graph))
        edge1 = rand(x)
        deleteat!(x, edge1)
        edge2 = rand(x)
        add_edge!(graph,edge1,edge2)
    end
    return graph
end

const MAX_ITER = 5000
const STOP_DIFF = 0.0001;

# a update(Newton Raphson)


#y-1./(1.+e.^-(a+a_0))+(u' * L)' + rho*L*(L*a-b)   


function gradient2(a,a_0,u,L,rho,b,y)
#    grad = grad+ (y[i]-invLogit(a+a_0))+(u' * L)[0:t_0] + rho*(L*a-append!(b, zeros(t-t_o,1)))
    grad = y-invLogit(a+a_0)+(u' * L)' - rho*L*(L*a-b)
    return grad
end;



function hessian(a,a_0,rho,L)
    hess = Diagonal(vec((invLogit(a+a_0).*(1-invLogit(a+a_0)))))+rho*L^2
    return -1*hess
end;





function newton(y_i,a_0,L,rho,b,u)
    a = zeros(length(y_i),1)
    a_old = a
    iters = 0
    diff = 1.0
    while(diff >STOP_DIFF && iters< MAX_ITER )
        grad = gradient2(a_old,a_0,u,L,rho,b,y_i)
        hess = hessian(a_old,a_0, rho,L)
        a = a_old - pinv(hess)*grad
        diff = norm(a-a_old)
        a_old = a
        iters = iters+1
    end
    return a
end

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



function ADMM(A,L,t,t_0,new)
	a = zeros(t,new)
	b = zeros(t)
	u = zeros(t,new)
#	alpha = 1.5  #relaxation parameter
	iters = 0
	diff = 1.0
	b_old = b
	while(diff >STOP_DIFF && iters< MAX_ITER )
		#a update
		for i in t_0:t
			a[1:length(A[i]),i] = newton(A[i],a_0,L[i],rho,b[1:length(A[i])],u[1:length(A[i]),i])
			u[1:length(A[i]),i] = u[1:length(A[i]),i]+ rho*(L[i]*a[1:length(A[i]),i]-b[1:length(A[i])])
		end
		c = zeros(t)
		for i in t_0:t
			c = c+ u[:,i]'+rho(L[i]*a[:,i])/((t-t_0)*rho/2)
		end
		b = sign(c).*max(abs(c)-lambda/2,0) #soft(c)
		diff  = norm(b-b_old)
		b_old = b
	end
	return b
end



######################################### testing

levels = 10
g = BinaryTree(levels)
n = nv(g)
b = (rand(n) .< 8 / n)*1. ;
g = randEdgeGen(g,1000)
A = Array{Int64,2}[]
numnewnodes = 5
for i in 1:numnewnodes
    g = addPrefNode(g,b)
    connects = zeros(2^levels-1+numnewnodes,1)  #-1 for -1 1 coding
    connects[neighbors(g,nv(g))] = 1
    push!(A,connects)
end

t = 2^levels-1+numnewnodes
t_0 = 2^levels-1
u = zeros(t,1);
rho = 1.1
lambda = 1.1
a_0 = -7
L = laplacian_matrix(g);
a = zeros(length(A[5]),1);
a_old = a;
iters = 0
diff = 1

#y-1./(1.+e.^-(a+a_0))+(u' * L)' + rho*L*(L*a-b)  


for i in 1:10
    grad = gradient2(a_old,a_0,u,L,rho,b,A[5])
    hess = hessian(a_old,a_0, rho,L)
    a = a_old - pinv(hess)*grad
    diff = norm(a-a_old)
    a_old = a
    iters = iters+1
end

println(a_old[1:5])
print(diff)