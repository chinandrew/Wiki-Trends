


#7345 ish
#12611 ishAC
#76674
#97070
#245300
include("graph_generation.jl")
using graph_generation
using LightGraphs
using JLD

max_seed = 1
max_nodes = 1
added = 0
for seed in 300000:350000
    try
        
        if added > max_nodes
            max_nodes = added
            max_seed = seed-1
        end
        print("SEED = ")
        println(seed)
        print("MAX SEED = ")
        println(max_seed)
        print(max_nodes)
        println("-------------------------------------------------")
        srand(seed);


        levels = 10  ;   #number of levels in binary tree
        new_edges = 0
        new_nodes = 500;
        side_length = 100
        a_0 = -4;


        #g = BinaryTree(levels);  #generate tree
        g = Grid([side_length,side_length])
        n = nv(g) ; #get initial number of nodes

        #b = (rand(n) .< 8 / n)*5. ;  #generate b vector

        #### SPARSE NORMAL
        b = zeros(n);
        nz = 30;
        nz_indices = rand(1:n,nz)
        for i in nz_indices
            b[i] = randn()
        end


        #### DENSE NORMAL
        b = randn(n)


        gen_b = copy(b);  # save for later
        g = graph_generation.randEdgeGen(g,new_edges);  
        A = Array{Int64,1}[];
        L =  SparseMatrixCSC{Int64,Int64}[];

        added = 0
        for i in 1:new_nodes 
            push!(L, laplacian_matrix(g));
            g = graph_generation.addPrefNode(g,b, a_0);
            #connects = zeros(2^levels-2+i)  #-1 for -1 1 coding;
            connects = zeros(side_length*side_length+i-1)  #-1 for -1 1 coding;
            connects[neighbors(g,nv(g))] = 1;
            push!(A,connects);
            println(i)
            added += 1
        end


        break

        while true
            print(seed)
        end 
    catch
        continue
    end
end


#save("graph_jld/Grid_100x100+0e+500n_sparse_seed1.jld","connections",A,"laplacians",L,"b" , gen_b,"a_0",a_0)

    