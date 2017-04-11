using LightGraphs
using DataFrames


data = readtable("training.csv")

t = maximum([maximum(data[:,4]),maximum(data[:,5])])


g = BinaryTree(0)

for i in 1:t_0
	add_vertex!(g)
end

for i in 1:size(data,1)
	add_edge!(g,data[i,4],data[i,5])
end
Lt = laplacian_matrix(g)


A = Array{Int64,2}[]
L =  SparseMatrixCSC{Int64,Int64}[]

new = 50
for i in reverse(1:new)
	n= size(Lt)[1]
	Ai = zeros(n,1)+(-1.*full(Lt[n-i+1,:]))
	Ai[n-i+1] = 0
	push!(A,Ai)
	push!(L, Lt[1:n-i,1:n-i])
end

