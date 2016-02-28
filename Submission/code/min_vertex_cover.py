# CSE 6140 Project
# Fall 2015
# Matthew Robinson, Allen Koh, Kenneth Droddy

import networkx as nx
import sys,os,getopt
import time
import itertools
from operator import itemgetter
import scipy.optimize as opt
import random
from math import floor

#--------------------------------------------------------------
# Procedures for importing .graph files
#---------------------------------------------------------------
def file_edges(filename):

    # Read the edges and weights in from the .graph file and arrange
    #   the data in a way that can be accepted by the networkx
    #   multi-graph object
    graph_file = open(filename,'r')
    # Parse the .graph file and create a list of edges and weights
    adj_list = []
    for line in graph_file:
        line = line.strip().split()
        # Convert input to int values
        data_list = [int(x) for x in line] 
        adj_list.append(data_list)
    graph_file.close()
    return adj_list

def read_graph(filename):
    adj_list = file_edges(filename)
    # The header contains data about the size of the graph
    header = adj_list[0]
    # Create an instance of the multi graph object and add edges from
    #   the adjacency list to the graph object
    G = nx.Graph()
    n = len(adj_list[1:])
    # For each node in the adjacency list, add edges to each node
    #   adjacent to it
    for i in range(n):
        current_node = i+1
        m = len(adj_list[i+1])
        for j in range(m):
            end_node = adj_list[i+1][j]
            G.add_edge(current_node,end_node)
    return G, header

#----------------------------------------------------------------------------
# Branch and Bound Code
#---------------------------------------------------------------------------
def nodes_by_degree(G):
    # Returns a list of the nodes in a graph, sorted by degree
    degree_list = sorted(G.degree_iter(),key=itemgetter(1),reverse=True)
    return degree_list

def residual_graph(G,v):
    # Input, G, the original graph
    # v, the vertex added to the vertex cover
    # degreeQ, the priority queue with node degrees
    #    from the original graph
    # Output: G', the graph consisting of edges not
    #    convered by C and the nodes not in C

    G1 = nx.Graph()
    for node in G.nodes():
        G1.add_node(node)
    for edge in G.edges():
        G1.add_edge(edge[0],edge[1])
    
    # Remove all edges in G that are covered by v
    neighbors = G1.neighbors(v)
    for u in neighbors:
        G1.remove_edge(v,u)
    # Remove v from G
    G1.remove_node(v)
        
    # Remove isolated nodes from G (this will include v)
    isolates = nx.isolates(G1)
    for node in isolates:
        G1.remove_node(node)
    #    degreeQ.remove_node(node)  
    return G1 

def vc_lp_relax(G):
    n = max(G.nodes())
    m = len(G.edges())

    # For the objective function, add a variable with
    #   coefficient 1 in the objective function for
    #   each node and create a slack variable with
    #   coefficient 0 in the objective function for
    #   each constraint (one for each edge)
    cvar = [1 for i in range(n)]
    cslack = [0 for i in range(m)]
    c = cvar + cslack

    # Add the constraints. In this case, xj+xi>=1 for
    #   each edge (i,j) in E
    b = [1 for i in range(m)]

    # Construct the matrix of constrains. Slack variables
    #   with coefficient -1 are added to make the constraints
    #   xj + xi >= 1
    A = []
    j = 0
    for edge in G.edges():
        const = [0 for i in range(n+m)]
        idx1 = edge[0] - 1
        idx2 = edge[1] - 1
        const[idx1] = 1
        const[idx2] = 1
        const[n+j] = -1
        A.append(const)
        j+=1
    # Run the linear program to find the lp relaxation solution,
    #   this is the lower bound for the ILP
    res = opt.linprog(c,A_eq=A,b_eq=b,bounds=(0,1))
    return res.fun

def minVC_BandB(G,C=None,unexplored=None,time_limit=3600):
    global best, best_cover, start_time, trace

    # Branch and Bounding algorithm for computing the minimum vertex
    #   cover for a graph G
    if C == None:
        C = []
        best = float('inf')
        trace = []
        start_time = time.time()

    total_time = time.time() - start_time
    if total_time > time_limit:
        return best_cover, trace

    degree_list = nodes_by_degree(G)

    if unexplored == None:
        unexplored = [x[0] for x in degree_list]

    # The minimum number of additional nodes is the solution to
    #   the LP relaxation of the ILP

    if G.nodes() != []:
        min_additional = vc_lp_relax(G)
    else:
        min_additional = 0

    if len(C) + min_additional >= best:
        return 'Fathom', trace
    if unexplored == []:
        if G.edges() == []:
            if len(C) < best:
                best = len(C)
                best_cover = C
                trace.append([len(C),total_time])
                return C, trace
            else:
                return 'Fathom', trace
        else:
            return 'Fathom', trace

    # Pick a node to branch on
    v = unexplored[0]
    unexplored.remove(v)

    # Branch with v in the covering set
    G1 = nx.Graph()
    for node in G.nodes():
        G1.add_node(node)
    for edge in G.edges():
        G1.add_edge(edge[0],edge[1])
    
    C1 = [x for x in C]
    C1.append(v)
    G1 = residual_graph(G1,v)
    residual_nodes = G1.nodes()
    unexplored1 = []
    for node in unexplored:
        if node in residual_nodes:
            unexplored1.append(node)
    soln1, trace = minVC_BandB(G1,C1,unexplored1)

    # Branch with v not in the covering set
    # Add all neighbors of v to the covering set,
    #   since it is necessary to conver all edges
    C2 = [x for x in C]
    
    G2 = nx.Graph()
    for node in G.nodes():
        G2.add_node(node)
    for edge in G.edges():
        G2.add_edge(edge[0],edge[1])
    

    # Create a residual graph that shows what
    #   still needs to be covered after all
    #   neighbors of v are added to the cover
    adj_nodes = G.neighbors(v)
    for node in adj_nodes:
        C2.append(node)
        G2 = residual_graph(G2,node)

    # Update the set of unexplored nodes
    residual_nodes = G2.nodes()
    unexplored2 = []
    for node in unexplored:
        if node in residual_nodes:
            unexplored2.append(node)
    soln2, trace = minVC_BandB(G2,C2,unexplored2)

    if soln1 == soln2 == 'Fathom':
        return 'Fathom', trace
    elif soln2 == 'Fathom':
        return soln1, trace
    elif soln1 == 'Fathom':
        return soln2, trace
    else:
        if len(soln1) <= len(soln2):
            return soln1, trace
        else:
            return soln2, trace

#--------------------------------------------------------------------
# Approximation
#---------------------------------------------------------------------

def minVC_approx(G):
    global start_time
    start_time = time.time()
    vertex_cover = []
    Gcopy = G.copy()
    # iterate until all edges covered
    while(Gcopy.number_of_edges() > 0):
        # get nodes of any remaining edge
        u,v = Gcopy.edges()[0]
        # add nodes to vertex cover
        vertex_cover.append(u)
        vertex_cover.append(v)
        # remove all edges adjacent to u or v
        Gcopy.remove_node(u)
        Gcopy.remove_node(v)
        
    trace=[[len(vertex_cover), time.time() - start_time]]
    return vertex_cover, trace

#---------------------------------------------------------------
# Local Search 1
#---------------------------------------------------------------

def minVC_LS1(G,restarts=20,pct=.25,randSeed=None):
    global start_time
    if randSeed != None:
        random.seed(randSeed)
    start_time = time.time()
    trace = []
    C = [node for node in G.nodes()]
    Best = [node for node in G.nodes()]
    for iteration in range(restarts+1):
        # If there are random restarts, insert a set percentage
        #   of the discarded nodes back into the vertex cover
        #   and run again
        if iteration != 0:
            discarded = [node for node in G.nodes() if node not in C]
            insertions = int( floor(pct*len(discarded)) + 1)
            for insertion in range(insertions):
                replace_node = random.choice(discarded)
                if replace_node not in C:
                    C.append(replace_node)
        edges = G.edges()
        random.shuffle(edges)
        for edge in edges:
            if (edge[0] in C) and (edge[1] in C):
                if G.degree(edge[0]) > G.degree(edge[1]):
                    if set(G.neighbors(edge[1])).issubset(set(C)):
                        C.remove(edge[1])
                        if len(C) < len(Best):
                            trace.append([len(C),time.time()-start_time])
                else:
                    if set(G.neighbors(edge[0])).issubset(set(C)):
                        C.remove(edge[0])
                        if len(C) < len(Best):
                            trace.append([len(C),time.time()-start_time])
        if len(C) < len(Best):
            Best = [node for node in C]
    return Best, trace

#-------------------------------------------------------------------------
# Local Search 2 - Hill Climbing
#-------------------------------------------------------------------------

#appx algorithm that serves only to give hill climbing a VC to start with
def minVC_approxHC(G):
    global start_time
    start_time = time.time()
    total_vertices = len(G.nodes())
    vertex_cover = []
    trace = []
    for u,v in G.edges():
        if (u in vertex_cover) or (v in vertex_cover):
            continue
        else:
            vertex_cover.append(u)
            vertex_cover.append(v)
            trace=[]            
            trace.append([len(vertex_cover),time.time() - start_time])
    return vertex_cover, trace

#hill climbing algorithm
def minVC_Hills(G,cutoffLS2=100,randSeedLS2=None):
    global start_time, bestLS
    R,_ = minVC_approxHC(G) #import approximation VC
    RY=R #copy VC vertices of R
    start_time=time.time()
    GL=G.copy()#copy input graph G, to be edited by algorithm later so that G is not tampered with
    xx=[]
    for k in range(0,len(RY)): #procedure to sort appx VC by degree
        k1=GL.degree(RY[k]) #procedure to sort appx VC by degree
        xx.append(k1) #procedure to sort appx VC by degree
    xxRY = zip(xx,RY) #procedure to sort appx VC by degree
    xxRY.sort() #procedure to sort appx VC by degree
    RZ = [RY for xx, RY in xxRY] #procedure to sort appx VC by degree
    RZ.reverse() #procedure to sort appx VC by degree
    random.seed(randSeedLS2)
    total_time=[]
    CL=[] #initialize array that will store V' solutions
    trace=[]
    lenR=len(R)
    bestLS = R #bestLS will represent new solution V', set equal to VC of appx algorithm so that if this algorithm times out before a new solution is found, it will return the VC we started with
    for k in range(0,len(R)):
        RL=RZ[k:len(R)] #VC from appx algorithm.  This loop starts with a target size of K and reduces the size by 1 on each iteration to try and find a smaller VC.
        GL=G.copy() #copy input graph G, to be edited by algorithm later so that G is not tampered with
        UE=GL.nodes() #compile list of nodes to consider
        #bestLS=[]
        CL=[]       
        random.shuffle(UE) #shuffle list of nodes in UE to randomize list of neighbors
        time_stop = time.time()-start_time
        if time_stop>cutoffLS2: #stop algorithm if cutoff time reached and record best solution found
            trace.append([len(bestLS),time.time()-start_time])
            break
        for z in range (0,len(RL)): #procedure to remove vertex from starting VC and look for better solution
            v=RL[z]
            UE=GL.nodes()
            if not UE: #if VC found, update solution and move to next iteration
                if (len(CL) < len(bestLS)):
                    bestLS=CL
                    trace.append([len(bestLS),time.time()-start_time])
                break
            if v in UE: #locate v in list of available vertices, v will be our starting vertex to try and climb from
                i = UE.index(v)
            else: #if v is no longer in list of available vertices, choose first available vertex in UE to explore
                v=UE[0]
                i = UE.index(v)
            for y in range (i,len(UE)): #search for neighbors of higher degrees
                if len(UE) == 1: #for case that only one vertex remains in the list of available vertices, return that vertex
                    v1=v
                    break
                elif y == len(UE)-1: #for case that we are at the end of the list and only one neighbor is available to check
                    if GL.degree(UE[y-1])>GL.degree(UE[y]):
                        v1=UE[y-1]
                        break
                    else:
                        v1=v
                        break       
                elif GL.degree(UE[y+1])>GL.degree(UE[y]): #check if neighbor with next highest index is better than current solution
                    v1=UE[y+1]
                    break
                elif y == 0:
                    v1=v
                    break
                elif GL.degree(UE[y-1])>GL.degree(UE[y]): #check if neighbor with next lowest index is better than current solution
                    v1=UE[y-1]
                    break
                elif GL.degree(UE[y])==GL.degree(UE[y+1]) and y==len(UE)-1: #this is implemented in the special case that all remaining vertices have degree 1, networkx was freezing during this situation, so this line was added to mitigate that issue.
                    v1=UE[y]
                    break
                else:
                    v1=UE[y] #if neighbors are not greater than v, return v'=v
            CL.append(v1) #add v1 or "v'" to the new VC, V'
            neighbors = GL.neighbors(v1)
            for q in neighbors: #remove edges connected to v'
                GL.remove_edge(v1,q)
            GL.remove_node(v1)
            isolates = nx.isolates(GL)
            for node in isolates: #remove nodes from list of available nodes if they have no edges connected to them
                GL.remove_node(node)
            time_stop = time.time()-start_time
            if time_stop>cutoffLS2: #stop algorithm if cutoff time reached and record best solution found
                trace.append([len(bestLS),time.time()-start_time])
                break
            if z == lenR-1:#this loop serves to make sure we've found a solution for a k-1 VC at a minimum, if we have not found one then the problem will reset and return to the first line of the algorithm
                if not GL.nodes():
                    if (len(CL) < len(bestLS)):
                        bestLS=CL
                        trace.append([len(bestLS),time.time()-start_time])
                elif k==1:
                   minVC_Hills(G) #this applies if a solution for k-1 not found; the algorithm is recalled and will continue until at least a k-1 solution is found
            if not GL.nodes(): #if VC found, update solution and move to next iteration
                if (len(CL) < len(bestLS)):
                    bestLS=CL
                    trace.append([len(bestLS),time.time()-start_time])
                break
    return bestLS,trace
    
#----------------------------------------------------------------------
# Run tests and collect data
#-----------------------------------------------------------------------

# Averaging method only used for finding the runtime across multiple runs
def ls_average(graph,method):
    output_dir = os.getcwd()+'/OUTPUT/'+str(method)+'/'
    file_list = [File for File in os.listdir(output_dir) if
                graph in File and '.trace' in File]
    i = 0
    time = 0
    cover = 0
    for trace_file in file_list:
        trace = []
        trace_file_data = open(output_dir+trace_file, 'r')
        for line in trace_file_data:
            line = line.strip().split(',')
            data_list = [float(x) for x in line]
            trace.append(data_list)
        time += trace[-1][0]
        cover += trace[-1][1]
        i+=1
    return time/i, cover/i   

# main execution method
def run_graph(inputFile,method,cutoff=600,randSeed=None):
    running_dir = os.getcwd()
    inputFilename = os.path.basename(inputFile)
    
    # Read in the input file
    G, header = read_graph(inputFile)
    
    # Start the timer
    start = time.time()
    
    # run the appropriate algorithm and produce output files
    if method == "BnB":

        # Run the algorithm
        cover, trace = minVC_BandB(G, time_limit = cutoff)
        # Stop the timer
        total_time = time.time() - start_time
        print "runtime: {}, number of vertices: {}".format(total_time, len(cover))
        # Produce comma separated string out of the vertex
        #   cover for use in the output file
        cover.sort()
        
        # Name the solution file
        soln_file = inputFilename +'_'+method+'_'+str(cutoff)+'.sol'
       
        # Name the trace file
        trace_file = inputFilename +'_'+method+'_'+str(cutoff)+'.trace'
    
    if method == "Approx":

        # Run the algorithm
        cover, trace = minVC_approx(G)
        # Stop the timer
        total_time = time.time() - start_time
        print "runtime: {}, number of vertices: {}".format(total_time, len(cover))
        # Produce comma separated string out of the vertex
        #   cover for use in the output file
        cover.sort()
        
        # Name the solution file
        soln_file = inputFilename +'_'+method+'_'+str(cutoff)+'.sol'
        
        # Name the trace file
        trace_file = inputFilename +'_'+method+'_'+str(cutoff)+'.trace'
    
    if method == "LS1": #edge by edge

        # Run the algorithm
        cover, trace = minVC_LS1(G,randSeed=randSeed)
        # Stop the timer
        total_time = time.time() - start_time
        print "runtime: {}, number of vertices: {}".format(total_time, len(cover))
        # Produce comma separated string out of the vertex
        #   cover for use in the output file
        cover.sort()
        
        # Name the solution file
        soln_file = inputFilename +'_'+method+'_'+str(cutoff)+'_'
        soln_file += str(randSeed)+'.sol'

        # Name the trace file
        trace_file = inputFilename +'_'+method+'_'+str(cutoff)+'_'
        trace_file += str(randSeed)+'.trace'

    if method == "LS2": #local hill climbing

        # Run the algorithm
        cover, trace = minVC_Hills(G,cutoffLS2=cutoff,randSeedLS2=randSeed)
        # Stop the timer
        total_time = time.time() - start_time
        print "runtime: {}, number of vertices: {}".format(total_time, len(cover))
        # Produce comma separated string out of the vertex
        #   cover for use in the output file
        cover.sort()

        # Name the solution file
        soln_file = inputFilename +'_'+method+'_'+str(cutoff)+'_'
        soln_file += str(randSeed)+'.sol'
        
        # Name the trace file
        trace_file = inputFilename +'_'+method+'_'+str(cutoff)+'_'
        trace_file += str(randSeed)+'.trace'

    # write output files
    cover_list = ','.join(map(str,cover))

    # solution file
    soln_output = open(running_dir + os.path.sep + soln_file, 'w')
    soln_output.write(str(len(cover))+'\n'+ cover_list)
    soln_output.close()

    # trace file
    trace_output = open(running_dir + os.path.sep + trace_file, 'w')
    for soln in trace:
        soln.reverse()
        soln_list = ','.join(map(str,soln))
        trace_output.write(soln_list+'\n')
    trace_output.close()


def main(argv):
    # defaults
    cutoff=600
    seed=None
    
    # read arguments
    try:
        opts, args = getopt.getopt(argv,"hi:a:t:s:", ["alg=","inst=","time=","seed="])
    except getopt.GetoptError:
        print "min_vertex_cover.py -i <inputGraphFile> -a <algorithm> -t <cutoff> -s <randomSeed>"
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print "min_vertex_cover.py -i <inputGraphFile> -a <algorithm> -t <cutoff> -s <randomSeed>"
            sys.exit()
        elif opt in ("-a", "--alg"):
            print "algorithm: {}".format(arg)
            if arg not in ("BnB", "Approx", "LS1","LS2"):
                print "alg must be one of the following: BnB, Approx, LS1, LS2"
                sys.exit()
            method = arg
        elif opt in ("-i","--inst"):
            input_graph_file = arg
        elif opt in ("-t", "--time"):
            cutoff = int(float(arg))
        elif opt in ("-s", "--seed"):
            seed = int(float(arg))

    # execute code
    run_graph(input_graph_file, method, cutoff, seed)

if __name__ == "__main__":
    main(sys.argv[1:])
