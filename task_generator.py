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

# Custom scripts.
from utils import Utils


class TaskGenerator:

    '''
    Generates compositional causal reasoning tasks.
    '''


    def __init__(self,
                 n_per_bcc: list = [3,3,3], 
                 bcc_types: list = ["cycle", "wheel", "cycle"],
                 bern: str = "uniform", # "random"
                 p: int = 0.5,
                 plot: bool = True):

        # For utility functions.
        self.utils = Utils()
        
        # Generate graphs.
        self.dag = self.get_dag(n_per_bcc = n_per_bcc,
                                bcc_types = bcc_types, 
                                plot = plot)
        self.adj_dag = self.get_adjacency_matrix(self.dag)
        self.nodes = list(self.dag.nodes())
        self.exog_names = [''.join(choices(string.ascii_uppercase+string.digits, k=4)) for _ in self.nodes]
        self.causal_functions = self.get_causal_functions()
        self.root = self.get_root(self.dag)
        self.leaf = self.get_leaf(self.dag)
        self.cutpoints = self.get_cutpoints(self.dag)
        self.cct_sort = [self.root] + self.cutpoints + [self.leaf]
        self.cct = self.get_cct(plot = False)

        # Enumerate quantities of interest.
        self.global_quantity = self.get_global()
        self.local = self.get_local()
        self.compositions = self.get_compositions()

        # Parameters to exogenous noise distributions.
        if bern == "random":
            self.p = np.random.uniform(low = 0.4, high = 0.8, size = len(self.nodes))
        else:
            self.p = [p]*len(self.nodes)


    def get_causal_functions(self) -> list:
        
        return ["or"]*(len(self.nodes))


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
    
        pass
    

    def get_cct(self,
                plot: bool = True) -> nx.classes.graph.Graph:

        '''
        Generates the commutative cut tree associated with the input causal DAG.
        '''

        self.adj_cct = np.triu(np.ones((len(self.cct_sort),len(self.cct_sort))), k = 1)
        self.adj_cct = self.adj_cct.astype(int)
        cct = nx.from_numpy_array(self.adj_cct, create_using = nx.DiGraph)
        cct = nx.relabel_nodes(cct, dict(zip(cct.nodes,self.cct_sort)))

        if plot:
            self.self.utils.plot_nx(self.adj_cct, 
                                    labels = self.cct_sort, 
                                    figsize = (7,7), 
                                    dpi = 50, 
                                    node_size = 1500,
                                    arrow_size = 20)
        return cct


    def get_cct_all_paths(self) -> list:

        '''
        Getter for composition cause-effect pairs for inductive CCR evaluation
        using Algorithm 1 / Theorem 1.

        Input is commutative cut tree (CCT), not the original causal DAG.
        '''
        
        return nx.all_simple_paths(self.cct, self.root, self.leaf)

    
    def get_cutpoints(self, 
                      dag: nx.classes.graph.Graph, 
                      topological_sort: bool = True) -> list:
        
        '''
        Getter for a topological sort of cutpoints.
        '''

        #nx.is_biconnected(dag.to_undirected())
        cutpoints = list(nx.articulation_points(dag.to_undirected()))
        if topological_sort:
            cutpoints = [x for x in dag.nodes if x in cutpoints]
        return cutpoints


    def get_leaf(self,
                 dag: nx.classes.graph.Graph):

        '''
        Getter for lone leaf in the graph.
        Returns node name.
        '''

        #leaf = [v for v, d in cct.out_degree() if d == 0][0]
        return list(dag.nodes())[-1]

    
    def get_root(self,
                 dag: nx.classes.graph.Graph,
                 return_name: bool = True):

        '''
        Getter for lone root in the graph.
        Returns node name.
        '''

        #root = [v for v, d in cct.in_degree() if d == 0][0]
        return list(dag.nodes())[0]


    def get_parents(self, 
                    var: str,
                    return_idx: bool = True) -> list:

        '''
        Get either the indices or names of the parents of a given node (var).
        '''

        var_idx = self.nodes.index(var)
        parents = list(np.nonzero(self.adj_dag[:,var_idx])[0])
        if return_idx:
            return parents
        return [self.nodes[i] for i in parents]


    def get_adjacency_matrix(self, 
                             dag: nx.classes.graph.Graph) -> np.ndarray:

        '''
        Getter for the numpy adjacency matrix.
        '''

        adj = nx.to_numpy_array(dag).astype(int)
        return np.triu(adj)
        #return nx.adjacency_matrix(dag)


    def get_cause_effect_pairs(self,
                               dag: nx.classes.graph.Graph) -> list:

        '''
        Getter for all relevant cause-effect pairs for inductive CCR evaluation
        using Algorithm 1 / Theorem 1.
        '''

        combos = list(itertools.combinations(self.cct_sort,2))
        
        # This step appears redundant, but I will keep this just in case 
        # the apparent sorting by itertools is not consistent.
        cause_effect_pairs = [x for x in combos if self.cct_sort.index(x[0]) < self.cct_sort.index(x[1])]
        return cause_effect_pairs


    def get_global(self) -> list:
        
        '''
        Getter for global quantity cause-effect pair for inductive CCR evaluation
        using Algorithm 1 / Theorem 1.
        '''

        return (self.root, self.leaf)


    def get_local(self) -> list:

        '''
        Getter for local quantity cause-effect pairs for inductive CCR evaluation
        using Algorithm 1 / Theorem 1.
        '''

        all_pairs = self.get_cause_effect_pairs(self.dag)
        return [x for x in all_pairs if x != (self.root, self.leaf)]


    def get_compositions(self) -> list:

        '''
        Getter for composition cause-effect pairs for inductive CCR evaluation
        using Algorithm 1 / Theorem 1.
        '''

        paths = self.get_cct_all_paths()
        compositions = []
        for path in paths:
            comp = [(path[i],path[i+1]) for i in range(len(path)-1)] 
            if len(comp) >= 2:
                compositions.append(comp)
        return compositions


    def sample_scm(self,
                   n: int = 1000,
                   intervene_node: str = None,
                   intervene_value: int = 0,
                   seed: int = 2024,
                   return_dfs: bool = True) -> pd.DataFrame:

        '''
        Sample from a structural causal model (SCM) with Bernoulli exogenous noise and 
        monotone boolean causal functions. Functions must be monotone to enable point
        identification of the probabilities of causation.

        Hard-coded example:
        A = noise_terms[0] if intervene_node != "A" else intervention()[0]
        B = fun((noise_terms[1],A)) if intervene_node != "B" else intervention()[0]
        C = fun((noise_terms[2],B)) if intervene_node != "C" else intervention()[0]
        D = fun((noise_terms[3],C)) if intervene_node != "D" else intervention()[0]
        E = fun((noise_terms[4],D)) if intervene_node != "E" else intervention()[0]
        Y = fun((noise_terms[5],E)) if intervene_node != "Y" else intervention()[0]
        '''

        # Returns [interventional sample, throwaway sample], where throwaway sample
        # is to ensure the random number generator stays consistent between the
        # observational and interventional dataframes (factuals and counterfactuals 
        # must be over the same exogenous variables for a valid joint).
        bern = lambda p: np.random.binomial(n = 1, p = p, size = n)
        if intervene_value:
            intervention = lambda : [np.ones(n).astype(int), bern(0.5)]
        else:
            intervention = lambda : [np.zeros(n).astype(int), bern(0.5)]

        # Sample Bernoulli exogenous noise.
        # Store noise terms so that factuals and counterfactuals are 
        # over the same exogenous variables. This is needed for a valid joint.
        np.random.seed(seed)
        noise_terms = [bern(self.p[i]) for i in range(len(self.nodes))]
        df_noise = pd.DataFrame(dict(zip(self.exog_names,noise_terms)))

        # Sample endogenous variables.
        # self.nodes should be in topological order, so parents will have been
        # generated before children (unless networkx changes its method).
        sample_dict = dict()
        for i in range(len(self.causal_functions)):
            if self.causal_functions[i] == "or":
                fun = lambda x: np.logical_or(x[0], x[1])
            else:
                fun = lambda x: np.logical_and(x[0], x[1])
            if intervene_node != self.nodes[i]:
                sample = noise_terms[i]
                #parents = list(dag.predecessors(node))
                parents_idx = np.nonzero(self.adj_dag[:,i])[0]
                parents = [self.nodes[j] for j in parents_idx]
                if len(parents) > 0:
                    for parent in parents:
                        sample = fun((sample,sample_dict.get(parent)))
                sample_dict[self.nodes[i]] = sample
            else:
                sample = intervention()[0]
                sample_dict[self.nodes[i]] = sample

        if return_dfs:
            return pd.DataFrame(sample_dict).astype(int), df_noise
        else:
            return sample_dict, dict(zip(self.exog_names,noise_terms))
    

    def get_truth(self, 
                  intervene_node: str = None,
                  intervene_value: int = 0) -> list:

        '''
        Get the ground truth for all endogenous variables (as binary vector)
        given the context prompt.
        '''
               
        self.endog_true_binary = [x for x in self.exog_true_binary]
        for i in range(len(self.nodes)):
            
            # Set logical operator.
            if self.causal_functions[i] == "and":
                fun = lambda x: int(np.logical_and(x[0], x[1]))
            elif self.causal_functions[i] == "or":
                fun = lambda x: int(np.logical_or(x[0], x[1]))
                
            # Get causal parents.
            parents_idx = np.nonzero(self.adj_dag[:,i])[0]

            # Generate data.
            if self.nodes[i] == intervene_node:
                self.endog_true_binary[i] = intervene_value
            else:
                for parent in parents_idx:
                    #print("var,fun,par:",self.endog_true_binary[i],self.causal_functions[i],self.endog_true_binary[parent])
                    self.endog_true_binary[i] = fun((self.endog_true_binary[i],self.endog_true_binary[parent]))
                    #print("=", self.endog_true_binary[i])
                        
        return self.endog_true_binary


    
