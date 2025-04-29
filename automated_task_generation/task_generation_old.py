# General importations.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import string
from random import shuffle,seed,choices
from faker.providers.person.en import Provider
import networkx as nx
import itertools

class TaskGenerator:


    def __init__(self,
                 n_per_bcc: list = [3,3,3], 
                 bcc_types: list = ["cycle", "wheel", "cycle"],
                 label_as: str = "names",
                 label_seed: int = None,
                 bern: str = "uniform", # "random"
                 p: int = 0.5,
                 plot: bool = True):

        '''
        label_as:
          - names: nodes given random human names
          - letters: node labels are uppercase letters in alphabetical order (by topological sort)
          - numbers: nodes labels are real numbers from [0..len(nodes)-1] (by topological sort)
          - nonsense: node labels are random uppercase strings (e.g., DRX, EOIDLZ, etc.)
        '''

        # Generate graphs.
        self.dag = self.get_dag(n_per_bcc = n_per_bcc,
                                bcc_types = bcc_types, 
                                label_as = label_as, 
                                label_seed = label_seed,
                                plot = plot)
        self.adj_dag = self.get_adjacency_matrix(self.dag)
        self.nodes = list(self.dag.nodes())
        self.root = self.get_root(self.dag)
        self.leaf = self.get_leaf(self.dag)
        self.cutpoints = self.get_cutpoints(self.dag)
        self.cct_sort = [self.root] + self.cutpoints + [self.leaf]
        self.cct = self.get_cct(plot = False)

        # Enumerate quantities of interest.
        self.global_quantity = self.get_global()
        self.local = self.get_local()
        self.compositions = self.get_compositions()

        # Quantitative thresholds / samples.
        if bern == "random":
            self.p = np.random.uniform(low = 0.2, high = 0.6, size = len(self.nodes))
        else:
            self.p = [p]*len(self.nodes)
        self.threshold = [int(x*10) for x in self.p]
        self.received = np.random.randint(low = 2, high = 8, size = len(self.nodes))

        # Qualitative desiderata / samples.
        self.init_colors()
        self.set_colors_wanted()
        self.set_colors_received()
        self.p_qualitative = [0.25]*len(self.nodes)


    def get_dag(self,
                n_per_bcc: list = [3,3,3], 
                bcc_types: list = ["cycle", "wheel", "cycle"],
                label_as: str = "names",
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
            - label_as: "names" labels nodes with randomly generated first names,
              "letters" as {A,B,C,...}, numbers" as {0,1,2,...}.
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
        if label_as == "names":
            #first_names = list(set(Provider.first_names))
            # Keep names all female just for now, for simplicity.
            first_names = list(set(Provider.first_names_female))
            if label_seed is not None:
                seed(label_seed)
            shuffle(first_names)
            labels = first_names[0:len(dag.nodes)]
        elif label_as == "letters":
            labels = [x for x in string.ascii_uppercase[:len(dag.nodes)]]
        elif label_as == "numbers":
            labels = range(len(dag.nodes))
        elif label_as == "nonsense":
            labels = [''.join(choices(string.ascii_uppercase+string.digits, k=4)) for _ in range(len(dag.nodes))]
        else:
            raise Exception("label_as must be in {names,letters,numbers,nonsense}.")
        dag = nx.relabel_nodes(dag, dict(zip(dag.nodes,labels)))
        dag = dag.to_directed(as_view = False)
    
        if plot:
            self.plot_nx(adj, 
                         labels = list(dag.nodes), 
                         figsize = (7,7), 
                         dpi = 50, 
                         node_size = 1500,
                         arrow_size = 20)
        return dag


    def get_cct(self,
                plot: bool = True) -> nx.classes.graph.Graph:

        '''
        Generates the commutative cut tree associated with the input causal DAG.
        '''

        self.adj_cct = np.triu(np.ones((len(self.cct_sort),len(self.cct_sort))), k = 1)
        cct = nx.from_numpy_array(self.adj_cct, create_using = nx.DiGraph)
        cct = nx.relabel_nodes(cct, dict(zip(cct.nodes,self.cct_sort)))

        if plot:
            self.plot_nx(self.adj_cct, 
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


    def get_adjacency_matrix(self, 
                             dag: nx.classes.graph.Graph) -> np.ndarray:

        '''
        Getter for the numpy adjacency matrix.
        '''

        adj = nx.to_numpy_array(dag).astype(int)
        return np.triu(adj)
        #return nx.adjacency_matrix(dag)


    def plot_nx(self,
                adjacency_matrix: np.ndarray,
                labels: list,
                figsize: tuple = (10,10),
                dpi: int = 200,
                node_size: int = 800,
                arrow_size: int = 10):

        '''
        Plot graph in networkx from adjacency matrix.
        '''
        
        g = nx.from_numpy_array(adjacency_matrix, create_using = nx.DiGraph)
        plt.figure(figsize = figsize, dpi = dpi)  
        nx.draw_circular(g, 
                         node_size = node_size, 
                         labels = dict(zip(list(range(len(labels))), labels)), 
                         arrowsize = arrow_size,
                         node_color = "pink",
                         with_labels = True)
        plt.show()
        plt.close()


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

    def init_colors(self):
        
        self.colors = ["lilac purple", "deep purple", "amethyst purple", "eggplant purple",
                       "pastel yellow", "canary yellow", "lemon yellow", "mustard yellow", 
                       "baby pink", "hot pink", "salmon pink", "bubblegum pink",
                       "fire engine red", "cardinal red", "blood red", "brick red"] # uniform probability. 
        self.color_families = ["purple", "yellow", "pink", "red"] # 1 in 4 chance = Bern(0.25) for happiness.


    def set_colors_wanted(self):
        
        if self.colors or self.color_families is None:
            self.init_colors()
            
        self.family_idx = np.random.randint(low = 0, high = len(self.color_families), size = len(self.nodes))
        self.colors_wanted = [self.color_families[idx] for idx in self.family_idx]

    
    def set_colors_received(self):

        if self.colors or self.color_families is None:
            self.init_colors()
        
        self.color_idx = np.random.randint(low = 0, high = len(self.colors), size = len(self.nodes))
        self.colors_received = [self.colors[idx] for idx in self.color_idx]


    def set_received(self, qualitative: bool = False):

        '''
        Setter for random numbers used in context prompts.
        '''

        if not qualitative:
            self.received = np.random.randint(low = 2, high = 9, size = len(self.nodes))
        else:
            self.set_colors_received()


    def sample_scm(self,
                   n: int = 1000,
                   #function: str = "or", # "and"
                   intervene_node: str = None,
                   qualitative: bool = False,
                   intervene_value: int = 0) -> pd.DataFrame:

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

        if self.conjunction is None:
            raise Exception("No conjunction has been selected. Execute generate_context_prompt() to remedy.")

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
        np.random.seed(2024)
        if not qualitative:
            noise_terms = [bern(self.p[i]) for i in range(len(self.nodes))]
        else:
            noise_terms = [bern(self.p_qualitative[i]) for i in range(len(self.nodes))]

        # Sample endogenous variables.
        # self.nodes should be in topological order, so parents will have been
        # generated before children (unless networkx changes its method).
        sample_dict = dict()
        for i in range(len(self.nodes)):
            if intervene_node != self.nodes[i]:
                sample = noise_terms[i]
                #parents = list(dag.predecessors(node))
                parents_idx = np.nonzero(self.adj_dag[:,i])[0]
                parents = [self.nodes[j] for j in parents_idx]
                if len(parents) > 0:
                    for parent in parents:
                        sample = self.fun((sample,sample_dict.get(parent)))
                sample_dict[self.nodes[i]] = sample
            else:
                sample = intervention()[0]
                sample_dict[self.nodes[i]] = sample
    
        return pd.DataFrame(sample_dict).astype(int)
    

    def generate_context_prompt(self, 
                                theme: str = "candy",
                                set_received: bool = False,
                                conjunction: str = "or") -> str:

        # Conjunction / causal function form.        
        self.conjunction = conjunction
        if self.conjunction == "and":
            conj = " " + self.conjunction + " her friend "
            self.fun = lambda x: np.logical_and(x[0], x[1])
        elif self.conjunction == "or":
            conj = " " + self.conjunction + " if "
            self.fun = lambda x: np.logical_or(x[0], x[1])
        else:
            raise Exception("Conjunction must be 'or' or 'and'.")
        
        if theme == "candy":
            intro = "A group of friends is going to a party where candies will be randomly distributed. "
            clauses = [" will be happy if she gets at least ", # clauses[0]
                       " candies",                             # clauses[1]
                       " is happy",                            # clauses[2]
                       " After distributing the candies, ",    # clauses[3]
                       " gets ",                               # clauses[4] 
                       " candies"]                             # clauses[5]
            if set_received:
                self.set_received()
        elif theme == "flowers":
            intro = "A group of friends is planting a bed of flowers from seed, but the seed packets are not labeled. "
            clauses = [" will be happy if the flowers she planted are ", # clauses[0]
                       " is happy",                                      # clauses[1]
                       " Once the flowers bloom, ",                      # clauses[2]
                       "'s flowers are "]                                # clauses[3] 
            if set_received:
                self.set_received(qualitative = True)
        elif theme == "vaccine":
            intro = "A group of friends is considering whether or not to get vaccinated against the flu this year. "
            clauses = [" will get vaccinated if she was sick for at least ", # clauses[0]
                       " days in the previous flu season",                   # clauses[1]
                       " gets vaccinated",                                   # clauses[2]
                       " During the previous flu season, ",                  # clauses[3]
                       " was sick for ",                                     # clauses[4]
                       " days"]                                              # clauses[5]
            if set_received:
                self.set_received()
        elif theme == "football":
            intro = "A group of friends is considering whether or not to attend an upcoming football game. "
            clauses = [" will go to the football game if she has been fewer than ", # clauses[0]
                       " times this season",                                        # clauses[1]
                       " goes to the game",                                         # clauses[2]
                       " So far this season, ",                                     # clauses[3]
                       " has been to ",                                             # clauses[4]
                       " games"]                                                    # clauses[5]   
            if set_received:
                self.set_received()
        else:
            raise Exception("Theme must be in {candy,flowers,vaccine,football}.")

        # Generate text.
        strings = [intro]
        if theme != "flowers":
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
            prompt = "".join(strings) + clauses[3]
            for name,number in zip(self.nodes[:len(self.nodes)-1],self.received[:len(self.nodes)-1]):
                prompt += name + clauses[4] + str(number) + clauses[5] + ", "
            prompt += "and " +  self.nodes[-1] + clauses[4] + str(self.received[-1]) + clauses[5] + "."
        else:
            for node,color in zip(self.nodes,self.colors_wanted):
                parents_idx = np.nonzero(self.adj_dag[:,self.nodes.index(node)])[0]
                parents = [self.nodes[i] for i in parents_idx]
                if len(parents) == 0:
                    string = node + clauses[0] + color + ". "
                else:
                    string = node + clauses[0] + color
                    for parent in parents:
                        string += conj + parent + clauses[1]
                    string += ". "
                strings.append(string)
            prompt = "".join(strings) + clauses[2]
            for name,color in zip(self.nodes[:len(self.nodes)-1],self.colors_received[:len(self.nodes)-1]):
                prompt += name + clauses[3] + color + ", "
            prompt += "and " +  self.nodes[-1] + clauses[3] + self.colors_received[-1] + "."
        
        self.context_prompt = prompt
        return prompt


    def get_ground_truth(self, 
                         qualitative: bool = False,
                         intervene_node: str = None,
                         intervene_value: int = 0) -> list:
        
        '''
        Compute ground truth.
        '''

        # Exogenous terms.
        if not qualitative:
            true = [x>=y for x,y in zip(self.received,self.threshold)]
            #print(true)
        else:
            true = [y in x for x,y in zip(self.colors_received,self.colors_wanted)]
            #print(true)            
            
        for i in range(len(self.nodes)):
            parents_idx = np.nonzero(self.adj_dag[:,i])[0]
            if len(parents_idx) > 0:
                for parent in parents_idx:
                    if self.nodes[parent] != intervene_node:
                        #print("no intervention")
                        true[i] = self.fun((true[i],true[parent]))
                    else:
                        #print("intervention")
                        true[i] = self.fun((true[i],intervene_value))
                    #print(self.nodes[i], ":", true)
        
        return true

    
    def generate_factual_prompt(self, 
                                effect_node: str,
                                theme: str = "candy") -> str:

        '''
        Generate prompt for the factual outcome.
        
        Params:
           - effect_node: node name corresponding to the effect in the cause-effect pair.
           - theme: prompt theme.

        Return: text prompt
        '''

        if theme in ["candy", "flowers"]:
            return "Is {} happy? Be as concise as possible.".format(effect_node)
        elif theme == "vaccine":
            return "Did {} get vaccinated? Be as concise as possible.".format(effect_node)
        elif theme == "football":
            return "Did {} go to the football game? Be as concise as possible.".format(effect_node)
        else:
            raise Exception("Theme must be in {candy,flowers,vaccine,football}.")
        

    def generate_counterfactual_prompt(self,
                                       effect_node: str,
                                       intervene_node: str = None,
                                       intervene_value: int = 0,
                                       theme: str = "candy") -> str:
        '''
        Generate prompt for the factual outcome.
        
        Params:
           - effect_node: node name corresponding to the effect in the cause-effect pair.
           - intervene_node: node name corresponding to the cause in the cause-effect pair.
           - intervene_value: value that cause node is fixed to, simulating intervention (in [0,1]).
           - theme: prompt theme.

        Return: text prompt
        '''

        if theme in ["candy", "flowers"]:
            if intervene_value:
                condition = "happy"
            else:
                condition = "not happy"
            prompt = "Now, suppose that {} is {}, regardless of her other circumstances. ".format(intervene_node,condition)
            prompt += "With this assumption, is {} happy? Be as concise as possible.".format(effect_node)
        elif theme == "vaccine":
            if intervene_value:
                condition = "got"
            else:
                condition = "did not get"
            prompt = "Now, suppose that {} {} vaccinated, regardless of her other circumstances. ".format(intervene_node,condition)
            prompt += "With this assumption, did {} get vaccinated? Be as concise as possible.".format(effect_node)
        elif theme == "football":
            if intervene_value:
                condition = "went"
            else:
                condition = "did not go"
            prompt = "Now, suppose that {} {} to the football game, regardless of her other circumstances. ".format(intervene_node,condition)
            prompt += "With this assumption, did {} go to the game? Be as concise as possible.".format(effect_node)
        else:
            raise Exception("Theme must be in {candy,flowers,vaccine,football}.")
        return prompt


