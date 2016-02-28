class priority_queue:
    
    # Implements a priority queue data structure using the Python heapq library
    # The priority queue keeps track of the node, its label and its priority
    # The data structure supports adding and deleting nodes in O(log n)
    #   and updating node labels in O(log n) by creating a dictionary that
    #   points to entries in the priority queue
    
    def __init__(self):
        self.queue = [] # Empty heap for the priority queue
        self.entry_dict = {} # Dictionary that points to entries in the queue
        self.removed = 'removed' # Label that indicates that a node was removed
        # Iterator to track the order that nodes were entered into the queue
        self.counter = itertools.count()
        
    def size(self):
        # Returns the size of the queue (for use in the while loop)
        return len(self.entry_dict)
        
    def add_node(self,node,priority):
        # Adds a node to the priority queue in O(log n)
        if node in self.entry_dict:
            self.remove_node(node)
        count = next(self.counter)
        entry = [priority, node]
        self.entry_dict[node] = entry
        heappush(self.queue,entry)
        
    def remove_node(self,node):
        # Removes a node to the priority queue in O(log n)
        entry = self.entry_dict.pop(node)
        entry[1] = self.removed
        
    def update_node(self,node,priority):
        # Updates the priority of a node in O(log n)
        self.remove_node(node)
        self.add_node(node, priority)
    
    def pop_node(self):
        # Pops the highest priority element in the queue
        while self.queue:
            priority, node = heappop(self.queue)
            if node is not self.removed:
                del self.entry_dict[node]
                return node, priority

def build_degree_queue(G):
    # From the graph, create a priority queue where the priority is
    #   the degree of the nodes in the graph
    # It takes O(|E||V|) to find the degrees of each node and build the queue
    degreeQ = priority_queue()
    for node in G.nodes():
        degreeQ.add_node(node,-G.degree(node))

    return degreeQ
