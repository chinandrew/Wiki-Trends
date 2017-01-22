#generate graph
levels = 10
g = BinaryTree(levels)
n = nv(g)
b = (rand(n) .< 8 / n)*1. ;
g = randEdgeGen(g,1000)
A = Array{Int64,2}[]
L =  SparseMatrixCSC{Int64,Int64}[]
numnewnodes = 5
for i in 1:numnewnodes
    g = addPrefNode(g,b)
    push!(L,laplacian_matrix(g))
    connects = zeros(2^levels-1+numnewnodes,1)  #-1 for -1 1 coding
    connects[neighbors(g,nv(g))] = 1
    push!(A,connects)
end	




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
		for i in 1:new
			a[1:length(A[i]),i] = newton(A[i],a_0,L[i],rho,b[1:length(A[i])],u[1:length(A[i]),i])
			u[1:length(A[i]),i] = u[1:length(A[i]),i]+ rho*(L[i]*a[1:length(A[i]),i]-b[1:length(A[i])])
		end
		c = zeros(t)
		for i in 1:numnewnodes
		    c[1:size(L[i])[1]] = c[1:1:size(L[i])[1]]+ u[1:size(L[i])[1],i]+rho*(L[i]*a[1:size(L[i])[1],i])/((t-t_0)*rho/2)
		end
		b = sign(c).*max(abs(c)-lambda/2,0) #soft(c)
		diff  = norm(b-b_old)
		b_old = b
	end
	return b
end

