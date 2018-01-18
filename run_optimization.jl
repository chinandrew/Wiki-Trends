@everywhere include("optimization.jl")
@everywhere using optimization
using JLD
    
invLogit(x) = 1./(1.+e.^-x)   
Logit(x) = log.(x./(1.-x))

#data = load("graph_jld/WS_10000-20-02b+0e+100n_normal*10_dense_seed1.jld");

data = load("graph_jld/Grid_100x100+0e+100n_normal+5.5_sparse_seed1.jld");

L = data["laplacians"];
A = data["connections"];
a_0 = data["a_0"];
gen_b = data["b"];

srand(1);

rho = 0.5
lambda = 0.009
pred_b = optimization.ADMM_grad_para(A,L,rho,lambda,a_0,0.0002);


total_pred_r = Array{Float64,2}(501,0);
total_gen_r = Array{Float64,2}(501,0);
total_pred_p = Array{Float64,2}(501,0);
total_gen_p = Array{Float64,2}(501,0);
SSE = 0 
bb = copy(gen_b);
pb = copy(pred_b);
for i =1:length(A)  
    pred_r = zeros(0)
    gen_r= zeros(0)
    pred_p = zeros(0)
    gen_p= zeros(0)
    pred_a = lufact(L[i])\(pb-mean(pb));
    pred_a -= mean(pred_a);
    gen_a = lufact(L[i])\(bb-mean(bb));
    gen_a -= mean(gen_a);
    SSE += sum((pred_a.-gen_a).^2)
    for k = 1:10:5001
        top_pred = sortperm(pred_a)[length(pred_a)-k:length(pred_a)];
        top_gen = sortperm(gen_a)[length(gen_a)-k:length(gen_a)];

        true_connects =find(A[i].==1);
        pred_rec = length(intersect(top_pred,true_connects))/length(true_connects);
        gen_rec = length(intersect(top_gen,true_connects))/length(true_connects);
        pred_prec = length(intersect(top_pred,true_connects))/(k+1)
        gen_prec = length(intersect(top_gen,true_connects))/(k+1)

        append!(gen_p,gen_prec)
        append!(pred_p, pred_prec)
        append!(pred_r, pred_rec)
        append!(gen_r, gen_rec)
    end
    println(string("node ", i, " complete"))
    total_pred_r = cat(2,total_pred_r, pred_r);
    total_gen_r = cat(2,total_gen_r, gen_r);
    total_pred_p = cat(2,total_pred_p, pred_p);
    total_gen_p = cat(2,total_gen_p, gen_p);
    append!(bb,0)
    append!(pb,0)
end

avg_pred_r = mean(total_pred_r,2)
avg_gen_r = mean(total_gen_r,2)
avg_pred_p = mean(total_pred_p,2)
avg_gen_p = mean(total_gen_p,2)
approx_random_r = collect(1:10:5001)./length(A[Int(length(A)/2)])
approx_random_p = (zeros(501,1)+sum(gen_b.!=0))./length(A[Int(length(A)/2)])
mean(avg_pred_r.-avg_gen_r)
mean(avg_pred_r.-approx_random_r)
mean(avg_gen_r.-approx_random_r)
SSE
length(find(pred_b.>0))
length(find(pred_b.!=0))
results = hcat(avg_pred_r,avg_gen_r,approx_random_r, avg_pred_p, avg_gen_p, approx_random_p);


writedlm("results/Grid_100x100+0e+100n_normal+6.5_sparse_seed1_05rho_009lambda.txt",results)


#writedlm("results/WS_10000-20-02b+0e+100n_normal*10_dense_seed1_05rho_0001lambda.txt",results)

    

writedlm("results/pred-b_Grid_100x100+0e+100n_normal+5.5_sparse_seed1_05rho_009lambda.txt",pred_b)
writedlm("results/Grid_100x100+0e+100n_normal+5.5_sparse_seed1_05rho_009lambda.txt",results)

    Grid_100x100+0e+100n_normal+7.5_sparse_seed1.jld


    
bb = copy(gen_b);
pb = copy(pred_b)
total_pred_r = []
total_gen_r = []
#for i =1:size(A)[1]
for i  = 1:10
    pred_a = lufact(L[i])\(pb-mean(pb));
    pred_a -= mean(pred_a)
    gen_a = lufact(L[i])\(bb-mean(bb));
    gen_a -= mean(gen_a);


    top_pred = sortperm(pred_a)[length(gen_a)-5000:length(gen_a)]
    top_gen = sortperm(gen_a)[length(gen_a)-5000:length(gen_a)]

    true_connects =find(A[i].==1)
    pred_rec = length(intersect(top_pred,true_connects))/length(true_connects)
    gen_rec = length(intersect(top_gen,true_connects))/length(true_connects)
    pred_prec = length(intersect(top_pred,true_connects))/
    gen_prec = length(intersect(top_gen,true_connects)) 

    println(gen_rec)
    println(pred_rec)
    println("--------")
    append!(bb,0)
    append!(pb,0)
    append!(total_pred_r, pred_rec)
    append!(total_gen_r, gen_rec)
end
println("##########")
println(mean(total_gen_r))
println(mean(total_pred_r))
