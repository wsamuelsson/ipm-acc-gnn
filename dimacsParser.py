import numpy as np
import graphviz
import scipy.sparse as spsp

def parseDimacs(file_path:str):
    with open(file=file_path, mode='r') as f:
        lines = f.readlines()
        arcs  = []
        objFunc = []
        arc_cc = 0
        for line in lines:
            parts = line.split()
            
            tp = parts[0]

            if tp == 'a':
                fromNode = parts[1]
                toNode = parts[2]
                lowerBound = parts[3]
                upperBound = parts[4]
                objFuncVal = parts[5]
                
                #Equality constraints
                #--------------------------------------- 
                #If we go from a node we want -1 at index (fromNode, arc)
                A_eq[int(fromNode)-1, arc_cc] = -1
                #If we go to a node we want +1 at index (toNode, arc)
                A_eq[int(toNode)-1, arc_cc] = 1
                
                #---------------------------------------
                #Inequality constraints
                #---------------------------------------
                b_ub[arc_cc] = int(upperBound)
                arc_cc += 1
                #---------------------------------------
                
                #Objective
                #---------------------------------------
                objFunc.append(int(objFuncVal))
                #---------------------------------------
                #For plotting 
                #---------------------------------------
                comment = f"    {lowerBound} <= capacity <= {upperBound}. Cost={objFuncVal}"
                arcs.append((fromNode, toNode, comment))
                #---------------------------------------
                
            if tp == 'n':
                nodeNum = int(parts[1])
                nodeSupply = int(parts[2])

                b_eq[nodeNum-1] = -nodeSupply
                nodes[nodeNum-1] = (nodeNum, nodeSupply)

            if tp == 'p':
                n_nodes = int(parts[2])
                n_arcs = int(parts[3])

                nodes = [(i+1, 0) for i in range(n_nodes)]
                b_eq = np.zeros(shape=(n_nodes, 1))
                A_eq = np.zeros(shape=(n_nodes, n_arcs))
                b_ub = np.zeros(shape=(n_arcs, 1))
                A_ub = spsp.eye(n_arcs, n_arcs, format='csr')

    #Remove first row for full rank matrix
    A_inc = A_eq.copy()
    A_eq, b_eq = A_eq[1:, :], b_eq[1:]

    #Compute memory for storing sparse matrices
    A_eq = spsp.csr_matrix(A_eq)
    mem_constraint_sparse = 2*(A_eq.nnz + A_ub.nnz) + A_eq.shape[0] + A_ub.shape[0] + len(objFunc)
    mem_constraint_sparse *= 4
    print(f"Using {mem_constraint_sparse / 1e6} MB of memory (CRS storage) for constraints and objective function.")

    return np.array(objFunc).reshape(-1,1), A_eq, b_eq, A_ub, b_ub, arcs, nodes, n_nodes, n_arcs, A_inc

import graphviz

def plot_graph(optimal_path, arcs, nodes, n_arcs, filename="graph", MAX_DENSITY=500):
    if n_arcs < MAX_DENSITY:
        g = graphviz.Digraph('network', format='pdf')  # Use PDF for LaTeX
        g.attr(dpi='300', rankdir='TB', nodesep='0.5', ranksep='0.7', fontname='Helvetica')

        # Add nodes with supply/demand labels
        for node in nodes:
            node_id = str(node[0])
            supply = int(round(node[1]))
            if supply >= 0:
                g.node(node_id, label=f"#{node_id}\\nSupply={supply}")
            else:
                g.node(node_id, label=f"#{node_id}\\nDemand={supply}")

        # Add edges with flow and metadata
        for i, arc in enumerate(arcs):
            u, v, label = arc[0], arc[1], arc[2]
            if optimal_path[i] > 0.1:
                flow = int(round(optimal_path[i].item()))
                g.edge(u, v, color='green', label=f"{flow} {label}", fontcolor="black")
            else:
                g.edge(u, v, color='red', label=label, fontcolor="red")

        # Output to file
        g.render(filename=filename, cleanup=True)
        print(f"Graph saved as {filename}.pdf")

def main():
    file_path = "net.dimacs"
    c, A_eq, b_eq, A_ub, b_ub, arcs, nodes, n_nodes, n_arcs, A_in = parseDimacs(file_path=file_path)
    
    

    
if __name__ == "__main__":
    main()