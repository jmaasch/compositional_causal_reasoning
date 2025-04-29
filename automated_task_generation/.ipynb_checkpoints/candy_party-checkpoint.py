# General importations.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import string
from random import shuffle,seed,choices
from faker import Faker
from faker.providers.person.en import Provider
import networkx as nx
import itertools
from utils import Utils

# Custom scripts.
from task_generator import TaskGenerator


class CandyParty(TaskGenerator):

    '''
    Generates compositional causal reasoning tasks.
    '''
        
    def get_dag(self,
                n_per_bcc: list = [3,3,3], 
                bcc_types: list = ["cycle", "wheel", "cycle"],
                label_seed: int = None,
                plot: bool = True) -> nx.classes.graph.Graph:
    
        '''
        Construct a directed acyclic graph (DAG) with exactly one root, exaclty one leaf, 
        varying numbers of biconnected components (BCCs), and varying numbers of nodes in 
        each BCC.

        Params:
            - n_per_bcc: list of number of nodes per BCC. 
            - bcc_types: list of graph structure type for each BCC with options 
              "cycle" (nx.cycle_graph) and "wheel" (nx.wheel_graph).
            - label_seed: random seed for name generator, if desired.
            - plot: show plot of DAG.

        Notes:
            1. n_per_bcc[i] >= 2.
            2. If n_per_bcc[i] == 2, bcc[i] will be a bridge.
            3. len(n_per_bcc) must equal len(bcc_types).
    
        Return: networkx digraph
        '''
    
        if len(n_per_bcc) != len(bcc_types):
            raise Exception("len(n_per_bcc) must be equal to len(bcc_types).")

        # Construct first BCC.
        if bcc_types[0] == "cycle":
            dag = nx.cycle_graph(n = n_per_bcc[0])
        elif bcc_types[0] == "wheel":
            dag = nx.wheel_graph(n = n_per_bcc[0])
        adj = nx.to_numpy_array(dag)
    
        # Convert adjacency matrix to upper triangular to get DAG.
        adj = np.triu(adj)
        dag = nx.from_numpy_array(adj) 
    
        # Get leaf.
        row_sums = adj.sum(axis = 1)
        leaf_idx = np.where(row_sums == 0)[0]
    
        # Add remaining BCCs.
        bccs = []
        for i in range(1,len(n_per_bcc)):
    
            if bcc_types[i] == "cycle":
                g = nx.cycle_graph(n = n_per_bcc[i])
            elif bcc_types[i] == "wheel":
                g = nx.wheel_graph(n = n_per_bcc[i])
    
            adj = nx.to_numpy_array(g)
            adj = np.triu(adj)
            g = nx.from_numpy_array(adj) 
            g = nx.relabel_nodes(g, dict(zip(list(g.nodes), [x+(len(dag.nodes)-1) for x in g.nodes])))
            
            dag = nx.relabel_nodes(dag, { n: str(n) if n==leaf_idx else 'a-'+str(n) for n in dag.nodes })
            g = nx.relabel_nodes(g, { n: str(n) if n==leaf_idx else 'b-'+str(n) for n in g.nodes })
            
            dag = nx.compose(dag,g)
            adj = nx.to_numpy_array(dag)
            adj = np.triu(adj)
            
            dag = nx.relabel_nodes(dag, dict(zip(list(dag.nodes), range(len(dag.nodes)))))
            row_sums = adj.sum(axis = 1)
            leaf_idx = np.where(row_sums == 0)[0]
    
        # Relabel nodes and cast as DiGraph.
        if label_seed is not None:
            seed(label_seed)
        first_names = list(set(Provider.first_names_female)) 
        shuffle(first_names)
        labels = first_names[0:len(dag.nodes)]
        dag = nx.relabel_nodes(dag, dict(zip(dag.nodes,labels)))
        dag = dag.to_directed(as_view = False)
        
        if plot:
            self.utils.plot_nx(adj, 
                               labels = list(dag.nodes), 
                               figsize = (7,7), 
                               dpi = 50, 
                               node_size = 1500,
                               arrow_size = 20)
        return dag


    def get_causal_context(self) -> str:

        '''
        Define causal model in text.
        '''

        intro = "A group of friends is going to a party where candies will be randomly distributed. "
        clauses = [" will be happy if she gets at least ", # clauses[0]
                   " candies",                             # clauses[1]
                   " is happy",                            # clauses[2]
                   " After distributing the candies, ",    # clauses[3]
                   " gets ",                               # clauses[4] 
                   " candies"]                             # clauses[5]

        strings = [intro]
        for node,number in zip(self.nodes,self.threshold):
            #parents = list(dag.predecessors(node)) # Auto-generated adjdagacency matrix is not upper triangular.
            #parents = dag.pred[node]
            parents_idx = np.nonzero(self.adj_dag[:,self.nodes.index(node)])[0]
            parents = [self.nodes[i] for i in parents_idx]
            if len(parents) == 0:
                string = node + clauses[0] + str(number) + clauses[1] + ". "
            else:
                string = node + clauses[0] + str(number) + clauses[1]
                for parent in parents:
                    string += conj + parent + clauses[2]
                string += ". "
            strings.append(string)
        self.causal_context = "".join(strings) + clauses[3]
        for name,number in zip(self.nodes[:len(self.nodes)-1],self.received[:len(self.nodes)-1]):
            self.causal_context += name + clauses[4] + str(number) + clauses[5] + ", "
        self.causal_context += "and " +  self.nodes[-1] + clauses[4] + str(self.received[-1]) + clauses[5] + "."

        return self.causal_context
        