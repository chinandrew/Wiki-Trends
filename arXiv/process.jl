@everywhere include("optimization.jl")
@everywhere using optimization
using LightGraphs
using DataFrames

data = readtable("training.csv");

t = maximum([maximum(data[:,5]),maximum(data[:,6])]);

g = BinaryTree(0);

for i in 1:t
    add_vertex!(g);
end
for i in 1:size(data,1)
    add_edge!(g,data[i,5],data[i,6]);
end

a = []
for i in 1:t
    push!(a, length(neighbors(g,i)))
end
sum(a.==0)

train_nodes = 100;
A = Array{Int64,1}[];
L = SparseMatrixCSC{Int64,Int64}[];

for i in 1:train_nodes
    connects = zeros(nv(g)-1);
    connects[neighbors(g,nv(g))] = 1;
    push!(A,connects);
    rem_vertex!(g,nv(g));
    push!(L, laplacian_matrix(g));
end

A = reverse(A);
L = reverse(L);

a_0 = -4;

rho = 0.5;
lambda = 0.005;
pred_b = optimization.ADMM_grad_para(A,L,rho,lambda,a_0,0.00005);
find(pred_b.>0)


pred_a = lufact(L[1])\(pred_b-mean(pred_b));
pred_a -= mean(pred_a);

top_pred = sortperm(pred_a)[length(pred_a)-50:length(pred_a)];