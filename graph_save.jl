
include("graph_generation.jl")
using graph_generation
using LightGraphs
using JLD

srand(1);


levels = 10  ;   #number of levels in binary tree
new_edges = 0
new_nodes = 500;
side_length = 100
a_0 = -4;


#g = BinaryTree(levels);  #generate tree
g = Grid([side_length,side_length])
n = nv(g) ; #get initial number of nodes

#b = (rand(n) .< 8 / n)*5. ;  #generate b vector

#### SPARSE NORMAL
b = zeros(n);
nz = 30;
nz_indices = rand(1:n,nz)
for i in nz_indices
    b[i] = randn()
end


#### DENSE NORMAL
#b = randn(n)


gen_b = copy(b);  # save for later
g = graph_generation.randEdgeGen(g,new_edges);  
A = Array{Int64,1}[];
L =  SparseMatrixCSC{Int64,Int64}[];


for i in 1:new_nodes 
    push!(L, laplacian_matrix(g));
    g = graph_generation.addPrefNode(g,b, a_0);
    #connects = zeros(2^levels-2+i)  #-1 for -1 1 coding;
    connects = zeros(side_length*side_length+i-1)  #-1 for -1 1 coding;
    connects[neighbors(g,nv(g))] = 1;
    push!(A,connects);
    println(i)
end


save("graph_jld/Grid_100x100+0e+500n_sparse_seed1.jld","connections",A,"laplacians",L,"b" , gen_b,"a_0",a_0)




#BinTree_10lvl+10000e+10n_sparse_seed1
#BinTree_10lvl+10000e+50_sparse_seed2
#BinTree_10lvl+10000e+100_sparse_seed3

#BinTree_13lvl+10000e+50n_sparse_seed4
#BinTree_13lvl+10000e+250n_sparse_seed5
#BinTree_13lvl+10000e+500n_sparse_seed6


#BinTree_10lvl+10000e+10n_dense_seed7
#BinTree_10lvl+10000e+50_dense_seed8
#BinTree_10lvl+10000e+100_dense_seed9

#BinTree_13lvl+10000e+50n_dense_seed10
#BinTree_13lvl+10000e+250n_dense_seed11
#BinTree_13lvl+10000e+500n_dense_seed12


