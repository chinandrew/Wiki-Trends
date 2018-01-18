using LightGraphs
using DataFrames
using Laplacians

data = readtable("validation.csv");

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

train_nodes = 99;
A = Array{Int64,1}[];
L = SparseMatrixCSC{Int64,Int64}[];

for i in 1:train_nodes
    connects = zeros(nv(g)-1);
    connects[neighbors(g,nv(g))] = 1;
    push!(A,connects);
    rem_vertex!(g,nv(g));
    push!(L, adjacency_matrix(g));
end

A = reverse(A);
L = reverse(L);


b = readdlm("results/ArXiv_300train_05rho_0005lambda_-4a0.txt")
b = vec(vcat(b,zeros((length(diag(L[1])) - length(b)))))




total_pred_r = Array{Float64,2}(501,0);
total_rank_r = Array{Float64,2}(501,0);
total_pred_p = Array{Float64,2}(501,0);
total_rank_p = Array{Float64,2}(501,0);


for i in 1:length(L)
    deg_ranks = reverse(sortperm(vec(sum(L[i],2))))
    f = cholLap(L[i])
    pred_a = vec(f((b-mean(b))))
    pred_a -= mean(pred_a)

    pred_r = zeros(0)
    rank_r= zeros(0)
    pred_p = zeros(0)
    rank_p= zeros(0)

    for k = 1:10:5001
        top_rank = deg_ranks[1:k]
        top_pred = reverse(sortperm(pred_a))[1:k]
        true_connects =find(A[i].==1);

        pred_rec = length(intersect(top_pred,true_connects))/length(true_connects);
        rank_rec = length(intersect(top_rank,true_connects))/length(true_connects);
        pred_prec = length(intersect(top_pred,true_connects))/k
        rank_prec = length(intersect(top_rank,true_connects))/k


        append!(rank_p, rank_prec)
        append!(pred_p, pred_prec)
        append!(pred_r, pred_rec)
        append!(rank_r, rank_rec)

    end
    println(string("node ", i, " complete"))
    total_pred_r = cat(2,total_pred_r, pred_r);
    total_rank_r = cat(2,total_rank_r, rank_r);
    total_pred_p = cat(2,total_pred_p, pred_p);
    total_rank_p = cat(2,total_rank_p, rank_p);
    append!(b,0)
end



avg_pred_r = mean(total_pred_r,2)
avg_rank_r = mean(total_rank_r,2)
avg_pred_p = mean(total_pred_p,2)
avg_gen_p = mean(total_gen_p,2)

