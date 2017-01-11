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
#	grad = grad+ (y[i]-invLogit(a+a_0))+(u' * L)[0:t_0] + rho*(L*a-append!(b, zeros(t-t_o,1)))
    grad = y-invLogit(a+a_0)+(u' * L)' + rho*L*(L*a-b)
	return grad
end;



function hessian(a,a_0,rho,L)
    hess = Diagonal(vec((invLogit(a+a_0).*(1-invLogit(a+a_0)))))+rho*L^2
    return -1*hess
end;





function newton(y_i,a_0,L,rho,b)
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
u = zeros(t,1)+0.2;
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
