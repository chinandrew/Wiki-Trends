module graph_generation


using LightGraphs
using StatsBase

invLogit(x) = 1./(1.+e.^-x)   

"""
given graph and probability, adds a node which must have >0
connections by flipping biased coin for each existing node
"""
function addNode(graph, p)
    add_vertex!(graph)
    x = nv(graph)
    degree = 0
    while degree ==0
        flips = rand(x-1)
        for i = 1:x-1
            if p[i]>flips[i]
                add_edge!(graph,i,x)
                degree +=1 
            end
        end
    end
    return graph
end



"""
given graph, b vector, and a_0, adds a new node as specified by the model
"""
function addPrefNode(g,b,a_0 = -7)
    n = nv(g)
    L::SparseMatrixCSC{Int64,Int64} = laplacian_matrix(g)
    a::Array{Float64,1} = lufact(L) \ (b - mean(b))
    p::Array{Float64,1} = invLogit(a+a_0)
    println(mean(p))
    addNode(g,p)
    push!(b,0)
    return g
end



"""
given graph and number of new edges desired, randomly adds edges between existing nodes
"""
function randEdgeGen(graph, newedges)
    for i in 1:newedges
        edges = sample(1:nv(graph), 2, replace = false)
        add_edge!(graph,edges[1],edges[2])
    end
    return graph
end


end
