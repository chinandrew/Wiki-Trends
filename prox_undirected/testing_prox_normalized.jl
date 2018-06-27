using LightGraphs
using DataFrames
using Laplacians
include("optimization_opt.jl")
using optimization_opt

data = readtable("validation_th_2.csv");

t = maximum([maximum(data[:,5]),maximum(data[:,6])]);

g = BinaryTree(0);

for i in 1:t
    add_vertex!(g);
end
for i in 1:size(data,1)
    add_edge!(g,data[i,5],data[i,6]);
end
 

test_nodes = 72;
A = Array{Int64,1}[];
L = SparseMatrixCSC{Float64,Int64}[];
D = SparseMatrixCSC{Float64,Int64}[];
a_0 = Array{Float64,1}[]

for i in 1:test_nodes
    connects = zeros(nv(g)-1);
    connects[neighbors(g,nv(g))] = 1;
    push!(A,connects);
    rem_vertex!(g,nv(g));

    lap = adjacency_matrix(g)


    #comb_lap =  laplacian_matrix(g)
    #x = ones(size(connects,1),2) 
    #x[:,2] = diag(comb_lap)/sum(diag(comb_lap))
    #y = connects
    #push!(a_0,vec(x*optimization_modified.newton(x,y)))



    Dsq = spdiagm(vec(sqrt.(sum(lap,2))))
    push!(L, lap);
    push!(D, Dsq)
end

A = reverse(A);
L = reverse(L);
D = reverse(D);
a_0 = reverse(a_0);


b = readdlm("results/ArXivTh_300train_1lambda_undirected_normalized_hybrid_connected_prox.txt")
b = vec(vcat(b,zeros((length(diag(L[1])) - length(b)))))


# 100 for patent, 99 for arxiv
for i in 1:test_nodes
    x = ones(size(A[i],1),2) 
    x[:,2] = sum(L[i],2)/(sum(L[i])/2)
    y = A[i]
    #push!(a_0,vec(x*optimization_modified.newton(x,y)))
    #push!(a_0,vec(x*[ -7.47907; 3479.19]))
    #push!(a_0,vec(x*[ -6.91616; 1857.95]))
    push!(a_0,vec(x*[ -6.10264; 885.314]))
end




# 100 for patent, 99 for arxiv
for i in 1:test_nodes
    x = ones(size(A[i],1),2) 
    x[:,2] = sum(L[i],2)/(sum(L[i])/2)
    y = A[i]
    #push!(a_0,vec(x*optimization_modified.newton(x,y)))
    #push!(a_0,vec(x*[ -7.47907; 3479.19]))
    #push!(a_0,vec(x*[ -6.91616; 1857.95]))
    push!(a_0,vec(x*[ -6.10264; 885.314]))
end




total_pred_r = Array{Float64,2}(100,0);
total_rank_r = Array{Float64,2}(100,0);
total_pred_p = Array{Float64,2}(100,0);
total_rank_p = Array{Float64,2}(100,0);


for i in 1:length(L)
    deg_ranks = reverse(sortperm(vec(sum(L[i],2))))
    f = cholLap(L[i])
    pred_a = vec(f( D[i]*(b-mean(b))))
    pred_a -= mean(pred_a)
    pred_a = D[i] * pred_a 
    pred_a = (D[i] * pred_a)  + 1.1*a_0[i]
    #pred_a = a_0[i]
    
    #pred_a = vec(sum(L[i],2)) / (sum(L[i])/2) + pred_a*0.0001
    #pred_a = vec(sum(L[i],2)) / (sum(L[i])/2)
    pred_r = zeros(0)
    rank_r= zeros(0)
    pred_p = zeros(0)
    rank_p= zeros(0)

    for k = 10:10:1000
        top_rank = deg_ranks[1:k]
        top_pred = reverse(sortperm(pred_a))[1:k]
        true_connects =find(A[i].==1);

        pred_rec = length(intersect(top_pred,true_connects))/length(true_connects)
        rank_rec = length(intersect(top_rank,true_connects))/length(true_connects)
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



avg_pred_r = mean(total_pred_r,2);
avg_rank_r = mean(total_rank_r,2);
avg_pred_p = mean(total_pred_p,2);
avg_rank_p = mean(total_rank_p,2);


results = hcat(avg_pred_r,avg_rank_r, avg_pred_p, avg_rank_p)

#writedlm("ArXivTh_300train_06lambda_undirected_hybrid_connected_prox_result.txt",results)

