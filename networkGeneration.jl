using LightGraphs

invLogit(x) = 1./(1.+e.^-x)    #define inverse logit


function addNode2(graph, p)
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


function randEdgeGen(graph, newedges)
    for i in 1:newedges
        x = collect(1:nv(graph))
        edge1 = rand(x)
        deleteat!(x, edge1)
        edge2 = rand(x)
        add_edge!(graph,edge1,edge2)
    end
    return graph
end

function addPrefNode(g,b,a_0 = -7)
    n = nv(g)
    L = laplacian_matrix(g)
    a = lufact(L) \ (b - mean(b))    
    p = invLogit(a+a_0)
    g = addNode2(g,p)
    push!(b,0)
    return g
end

function pastNeighbors(g,v)
    N = neighbors(g,v)
    return N[N < v]



levels = 12
g = BinaryTree(levels)
n = nv(g)
b = (rand(n) .< 8 / n)*1. 
g = randEdgeGen(g,1000)
A = Array{Int64,2}[]
for i in 1:500
    g = addPrefNode(g,b)
    connects = zeros(nv(g))
    connects[neighbos(g,nv(g))] = 1
    push!(A,connects)
end
