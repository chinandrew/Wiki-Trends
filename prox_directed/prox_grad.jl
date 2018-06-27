iters = 100


soft(c,lambda) = sign.(c).*max.(abs.(c)-lambda/2,0)


invLogit(x) = 1./(1.+e.^-x) 

function pseudo(b,y)
    a = zeros(length(y))
    for i in 1:length(y)
        a[i] = b[i] + mean( a[y.==1])
    end
    return (a - mean(a))
end


lambda = 0.7
step = 0.001

tic()

T = length(A[length(A)]);
t0 = length(A[1]);
b = 1*(rand(t0).<0.01);
#b = append!(b, zeros(T-t0))
for i in 1:iters
    temp = zeros(t0);
    lb = pseudo( vcat(b,zeros(T-t0)) , A[length(A)]) ;
    for t in 1:length(A)
        temp += (L[t]* (invLogit(lb[1:(t0+t-1)] +a_0[t]) - A[t] ))[1:t0];
    end
	prox = b - step * temp;
	b = soft(prox,lambda);
    println("iters: $(i)")
    #println(sum(b.!=0))
    #println(b[find(b.>0)])
   # println(sum(b.>0))
end

toc()

find(b.>0)
writedlm("results/ArXivTh_300train_07lambda_normalized_hybrid_connected_prox.txt",b)
