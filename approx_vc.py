import networkx as nx
import sys,os,getopt
import time

#--------------------------------------------------------------
# Procedures for importing .graph files
#---------------------------------------------------------------
def file_edges(filename):
    #---------------------------------------------------------
    directory = os.getcwd()+'/DATA/'
    #----------------------------------------------------------
    
    filename = directory +filename
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
# Approximation Code
#---------------------------------------------------------------------------
# get all edges from input graph
# loop through each edge
#   add to vertex cover graph if u,v of current edge not yet covered
# should be maximum < 2*optimal
def approximate(G, start_time):
    total_vertices = len(G.nodes())
    vertex_cover = []
    trace = []
    for u,v in G.edges():
        if (u in vertex_cover) or (v in vertex_cover):
            continue
        else:
            vertex_cover.append(u)
            vertex_cover.append(v)
            trace.append([len(vertex_cover)),time.time() - start_time])
    return vertex_cover, trace

#----------------------------------------------------------------------
# Run tests and collect data
#-----------------------------------------------------------------------

def run_graph(inputFile,method,cutoff=600,randSeed=None):
    # Read in the input file
    G, header = read_graph(inputFile)

    # run the appropriate algorithm and produce output files
    if method == "approx":

        # Start the timer
        start_time = time.time()
        # Run the algorithm
        cover, trace = approximate(G, start_time)
        # Stop the timer
        total_time = time.time() - start_time
        print "runtime: {}, number of vertices: {}".format(total_time, len(cover))
        # Produce comma separated string out of the vertex
        #   cover for use in the output file
        cover.sort()
        cover_list = ','.join(map(str,cover))
        
        # Set up and write to the output files
        output_dir = os.getcwd()+'/OUTPUT/'

        # Write the solution file        
        soln_file = inputFile +'_'+method+'_'+str(cutoff)+'.sol'
        soln_output = open(output_dir + soln_file, 'w+')
        soln_output.write(str(len(cover))+'\n'+ cover_list)
        soln_output.close()

         # Write the trace file        
        trace_file = inputFile +'_'+method+'_'+str(cutoff)+'.trace'
        trace_output = open(output_dir + trace_file, 'w')
        for soln in trace:
            soln_list = ','.join(map(str,soln))
            trace_output.write(soln_list+'\n')
        trace_output.close()


def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hm:i:")
    except getopt.GetoptError:
        print "approx_vc.py -m <method> -i <inputGraphFile>"
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print "approx_vc.py -m <method> -i <inputGraphFile>"
            sys.exit()
        elif opt == '-m':
            print "method: {}".format(arg)
            method = arg
        elif opt == "-i":
            input_graph_file = arg

    run_graph(input_graph_file, method)

if __name__ == "__main__":
    main(sys.argv[1:])