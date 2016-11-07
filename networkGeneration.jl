using LightGraphs

invLogit(x) = 1./(1.+e.^-x)    #define inverse logit
graph = [2 -1 0 0 -1 0 ; -1 3 -1 0 -1 0 ; 0 -1 2 -1 0 0 ; 0 0 -1 3 -1 -1; -1 -1 0 -1 3 0;0 0 0 -1 0 1];    #laplacian matrix
b = [ 0 1 1 0 0 0]';
a = pinv(graph)*b;
p = invLogit(a+log(1/10));    #creates vector of probabilities
L = sparse(graph)

function newNode(graph, p)
    degree = 0
    graph = hcat(graph, zeros(Int,size(graph)[1]))    #add new row/column representing new node
    graph =  vcat(graph,zeros(Int, size(graph)[2])')
    while degree == 0    #ensure at least one attachment
        flips = rand(size(graph)[1])
        for i = 1:size(graph)[1]-1    #determine attachment or not
            if p[i]>flips[i]
                degree +=1
                graph[size(graph)[1],i] = -1
                graph[i,size(graph)[1]] = -1
            end
        end
        graph[size(graph)[1],size(graph)[1]]  = degree    #add node degree to diagonal
    end 
    return graph
end

invLogit(x) = 1./(1.+e.^-x)

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


levels = 10
g = BinaryTree(levels)
n = nv(g)
b = (rand(n) .< 8 / n)*1. 

function addPrefNode(g,b,a_0 = -7)
    n = nv(g)
    L = laplacian_matrix(g)
    a = lufact(L) \ (b - mean(b))    
    p = invLogit(a+a_0)
    addNode2(g,p)
    push!(b,0)
    return p
end

function pastNeighbors(g,v)
    N = neighbors(g,v)
    return N[N < v]

function randEdgeGen(graph, newedges)
    for i in 1:newedges
        z = newedges
        x = collect(1:nv(graph))
        edge1 = rand(x)
        deleteat!(x, edge1)
        edge2 = rand(x)
        add_edge!(graph,edge1,edge2)
    end
    return graph
end

=======

