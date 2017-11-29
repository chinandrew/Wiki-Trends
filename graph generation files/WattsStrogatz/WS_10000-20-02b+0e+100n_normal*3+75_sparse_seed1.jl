


#7345 ish
#12611 ishAC
#76674
#97070

include("graph_generation.jl")
using graph_generation
using LightGraphs
using JLD

max_seed = 1
max_nodes = 1
added = 0

srand(1);


levels = 10  ;   #number of levels in binary tree
new_edges = 0
new_nodes = 100
a_0 = -6.5;
#a_0 = -6; 


#g = BinaryTree(levels);  #generate tree
g = watts_strogatz(10000,20,0.2)
n = nv(g) ; #get initial number of nodes

#b = (rand(n) .< 8 / n)*5. ;  #generate b vector

#### SPARSE NORMAL
b = zeros(n);
nz = 30;
nz_indices = rand(1:n,nz)
for i in nz_indices
    b[i] = randn()*3+75
end


#### DENSE NORMAL
#b = randn(n)


gen_b = copy(b);  # save for later
g = graph_generation.randEdgeGen(g,new_edges);  
A = Array{Int64,1}[];
L =  SparseMatrixCSC{Int64,Int64}[];

added = 0
for i in 1:new_nodes 
    push!(L, laplacian_matrix(g));
    g = graph_generation.addPrefNode(g,b, a_0);
    #connects = zeros(2^levels-2+i)  #-1 for -1 1 coding;
    connects = zeros(n+i-1)  #-1 for -1 1 coding;
    connects[neighbors(g,nv(g))] = 1;
    push!(A,connects);
    println(i)
    added += 1
end



save("graph_jld/WS_10000-20-02b+0e+100n_normal*3+75_sparse_seed1.jld","connections",A,"laplacians",L,"b" , gen_b,"a_0",a_0)


gen_a = lufact(L[1])\(gen_b-mean(gen_b));
gen_a -= mean(gen_a);
invLogit(x) = 1./(1.+e.^-x)   ;
gen_p = invLogit(gen_a+a_0)
gen_p[find(b.>0)]
mean(gen_p)

