invLogit(x) = 1./(1.+e.^-x)    #define inverse logit
graph = [2 -1 0 0 -1 0 ; -1 3 -1 0 -1 0 ; 0 -1 2 -1 0 0 ; 0 0 -1 3 -1 -1; -1 -1 0 -1 3 0;0 0 0 -1 0 1];    #laplacian matrix
b = [ 0 1 1 0 0 0]';
a = pinv(graph)*b;
p = invLogit(a+log(1/10));    #creates vector of probabilities


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