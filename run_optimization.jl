@everywhere include("optimization.jl")
@everywhere using optimization
using JLD

invLogit(x) = 1./(1.+e.^-x)   
Logit(x) = log.(x./(1.-x))

data = load("graph_jld/WS_10000-20-02b+0e+100n_normal*3+75_sparse_seed1.jld");
L = data["laplacians"];
A = data["connections"];
a_0 = data["a_0"];
gen_b = data["b"];

srand(1)

rho = 0.5
lambda = 0.0022
pred_b = optimization.ADMM_grad_para(A,L,rho,lambda,a_0,0.0005);


total_pred = Array{Float64,2}(501,0);
total_gen = Array{Float64,2}(501,0);
SSE = 0 
bb = copy(gen_b);
pb = copy(pred_b);
for i =1:length(A)  
    pred = zeros(0)
    gen = zeros(0)
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
        append!(pred, pred_rec)
        append!(gen, gen_rec)
    end
    println(string("node ", i, " complete"))
    total_pred = cat(2,total_pred, pred);
    total_gen = cat(2,total_gen, gen);
    append!(bb,0)
    append!(pb,0)
end

avg_pred = mean(total_pred,2)
avg_gen = mean(total_gen,2)
approx_random = collect(1:10:5001)./length(A[Int(length(A)/2)])
mean(avg_pred.-avg_gen)
mean(avg_pred.-approx_random)
mean(avg_gen.-approx_random)
SSE
length(find(pred_b.>0))
length(find(pred_b.!=0))
results = hcat(avg_pred,avg_gen,approx_random);


writedlm("results/WS_10000-20-02b+0e+100n_normal*3+100_sparse_seed1_05rho_00034lambda.txt",results)

    


    
bb = copy(gen_b);
pb = copy(pred_b)
total_pred = []
total_gen = []
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

    println(gen_rec)
    println(pred_rec)
    println("--------")
    append!(bb,0)
    append!(pb,0)
    append!(total_pred, pred_rec)
    append!(total_gen, gen_rec)
end
println("##########")
println(mean(total_gen))
println(mean(total_pred))
