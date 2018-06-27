@everywhere include("optimization_modified.jl")
@everywhere using optimization_modified
using LightGraphs
using DataFrames

data = readtable("training_th_2.csv");

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

train_nodes = 300;
A = Array{Int64,1}[];
L = SparseMatrixCSC{Float64,Int64}[];
D = SparseMatrixCSC{Float64,Int64}[];
Adj = SparseMatrixCSC{Float64,Int64}[];

max_neighbor =  Array{Float64,1}[];
#a_0 = Array{Float64,1}[]



X = Array{Float64,2}(0,2)
Y = Array{Float64,1}(0)

individual_X =  Array{Float64,2}[]


for i in 1:train_nodes
    connects = zeros(nv(g)-1);
    connects[neighbors(g,nv(g))] = 1;
    neighbor_deg = Int64[]
    for i in 1:nv(g)
        push!(neighbor_deg, maximum([length(neighbors(g,j)) for j in neighbors(g,i)]))
    end
    push!(max_neighbor,neighbor_deg)
   # push!(max_neighbor,[maximum(neighbors(g,i)) for i in 1:nv(g)])
    push!(A,connects);
    rem_vertex!(g,nv(g));

    adjmat = adjacency_matrix(g)
    lap = laplacian_matrix(g)
    Dsq = spdiagm(vec(sqrt.(sum(adjmat,2))))
    x = ones(size(connects,1),2) 
    x[:,2] = diag(lap)/sum(diag(lap))
    push!(individual_X, x)
    X = vcat(X,x)

    Y = vcat(Y,connects)

    #x = ones(size(connects,1),2) 
    #x[:,2] = diag(lap)/sum(diag(lap))
    #y = connects
    #push!(a_0,vec(x*optimization_modified.newton(x,y)))

    Dsq_inv = spdiagm(1./sqrt.(diag(lap)))
    normlap = Dsq_inv * lap * Dsq_inv
    push!(Adj, adjmat);
    push!(L, normlap);
    push!(D, Dsq)
end



A = reverse(A);
L = reverse(L);
D = reverse(D);
Adj = reverse(Adj);

max_neighbor =reverse(max_neighbor);
#a_0 = reverse(a_0);
individual_X = reverse(individual_X);

coefs = optimization_modified.newton(X,Y);
a_0 = Array{Float64,1}[]

for x in individual_X
    push!(a_0,vec(x*coefs))
end


rho = 0.5;
lambda = 0.07;
