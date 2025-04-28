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
        self.causal_functions = ["or"]*(len(self.nodes)-1)+["and"]
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
            self.p = np.random.uniform(low = 0.2, high = 0.6, size = len(self.nodes))
        else:
            self.p = [p]*len(self.nodes)


    def get_dag(self,
                n_per_bcc: list = [3,3,3], 
                bcc_types: list = ["cycle", "wheel", "cycle"],
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
        labels = [''.join(choices(string.ascii_uppercase+string.digits, k=4)) for _ in range(len(dag.nodes)-2)]
        labels = ["pain"]+labels+["surgery"]
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
    

    def get_cct(self,
                plot: bool = True) -> nx.classes.graph.Graph:

        '''
        Generates the commutative cut tree associated with the input causal DAG.
        '''

        self.adj_cct = np.triu(np.ones((len(self.cct_sort),len(self.cct_sort))), k = 1)
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
        np.random.seed(2024)
        noise_terms = [bern(self.p[i]) for i in range(len(self.nodes))]
        df_noise = pd.DataFrame(dict(zip(self.exog_names,noise_terms)))

        # Sample endogenous variables.
        # self.nodes should be in topological order, so parents will have been
        # generated before children (unless networkx changes its method).
        sample_dict = dict()
        for i in range(len(self.nodes)):
            if i == len(self.nodes)-1:
                fun = lambda x: np.logical_and(x[0], x[1])
            else:
                fun = lambda x: np.logical_or(x[0], x[1])
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
    

    def get_causal_context(self) -> str:
    
        # Get variable metadata for context prompt.
        self.var_dict = dict()
        self.alleles = []             # exogenous variables.
        self.family_history = []      # exogenous variables.
        self.previous_surgeries = []  # exogenous variables.
        self.disease = ''.join(choices(string.ascii_uppercase+string.digits, k=6))
        self.pain_threshold = np.random.choice([7, 8, 9], size = 1).item()
        endog_options = ["lab", "vital"]
        exog_options = ["carries allele", "has a family history of", "has previously received surgery for"]
        magnitudes = ["elevated", "low"]

        for var,u in zip(self.nodes,self.exog_names):
        
            parents = self.get_parents(var, return_idx = False)
            if var == "pain":
                exog = "carries allele"
                endog = None
            elif var == "surgery":
                exog = np.random.choice(exog_options, size = 1).item()
                endog = None
            else:
                exog = np.random.choice(exog_options, size = 1).item()
                endog = np.random.choice(endog_options, size = 1).item()
        
            # Store exogenous variables by type.
            if "allele" in exog:
                self.alleles.append(u)
            elif "family" in exog:
                self.family_history.append(u)
            elif "surgery" in exog:
                self.previous_surgeries.append(u)
        
            # Get magnitudes.
            mag = np.random.choice(magnitudes, size = 1).item()
            level = str(round(np.random.uniform(low = 0.1, high = 3.5, size = 1).item(), 2))+" mg/dL"
            if mag == "low":
                level = "less than "+level
            else:
                level = "greater than "+level
                
            self.var_dict[var] = {"parents": parents,
                                  "endog type": endog, 
                                  "endog magnitude": mag, 
                                  "endog level": level,
                                  "exog var name": u, 
                                  "exog type": exog}

        # Construct prompt.
        intro = "Chronic disease {} sometimes requires surgical intervention,".format(self.disease)
        intro += " depending on genetics, patient history, vital signs, and lab results. The patient will experience" 
        intro += " significant pain (rated greater than or equal to {}/10)".format(self.pain_threshold)
        intro += " if they carry allele {},".format(self.var_dict.get("pain").get("exog var name"))
        intro += " a genetic marker for severe {}.".format(self.disease)
        outro = "Assume that all factors influencing the surgeon are fully described here."
        strngs = [intro]
        for var,terms in self.var_dict.items():
            if var == "pain":
                continue
            parents = terms.get("parents")
            endog_type = terms.get("endog type")
            magnitude = terms.get("endog magnitude")
            level = terms.get("endog level")
            exog = "the patient "+terms.get("exog type")+" "+terms.get("exog var name")
        
            parent_strngs = []
            for parent in parents:
                if parent == "pain":
                    parent = "the patient self-reports significant pain"
                else:
                    parent_terms = self.var_dict.get(parent)
                    parent = "{} is {}".format(parent,parent_terms.get("endog magnitude"))
                parent_strngs.append(parent)
            parent_strngs.append(exog)
            if var != self.leaf:
                if var == self.root:
                    strng = " and ".join(parent_strngs)
                else:
                    strng = " or ".join(parent_strngs)
                strng = "If " + strng + ", then {} {} will be {} ({}).".format(endog_type,
                                                                               var,
                                                                               magnitude,
                                                                               level)
            else:
                strng = " and ".join(parent_strngs)
                strng = "If " + strng + ", then the surgeon will recommend surgery.".format(endog_type,
                                                                                            var,
                                                                                            magnitude,
                                                                                            level)
            strngs.append(strng)
        strngs.append(outro)
        
        self.causal_context = " ".join(strngs)
        return self.causal_context


    def get_patient_history(self,
                            n_extra_vars: int = 2) -> str:

        # Get patient sex and name according to sex.
        self.sex = np.random.choice(["male", "female"], size = 1).item()
        f = Faker()
        if self.sex == "female":
            self.name = f.name_female()
        else:
            self.name = f.name_male()
        
        # Get observed alleles.
        if len(self.alleles) > 0:
            total_alleles = np.random.randint(low = 1, high = len(self.alleles)+1, size = 1).item()
        else:
            total_alleles = 0
        self.obs_alleles = list(np.random.choice(self.alleles, size = total_alleles))
        self.extra_alleles = ["".join(choices(string.ascii_uppercase+string.digits, k=4)) for _ in range(n_extra_vars)]
        obs_alleles_str = ", ".join(self.obs_alleles+self.extra_alleles)
        
        # Get observed family medical history.
        if len(self.family_history) > 0:
            total_family_history = np.random.randint(low = 1, high = len(self.family_history)+1, size = 1).item()
        else:
            total_family_history = 0
        self.obs_family_history = list(np.random.choice(self.family_history, size = total_family_history))
        self.extra_family_history = ["".join(choices(string.ascii_uppercase+string.digits, k=4)) for _ in range(n_extra_vars)]
        obs_family_history_str = ", ".join(self.obs_family_history+self.extra_family_history)
        
        # Get observed surgical history.
        if len(self.previous_surgeries) > 0:
            total_previous_surgeries = np.random.randint(low = 1, high = len(self.previous_surgeries)+1, size = 1).item()
        else:
            total_previous_surgeries = 0
        self.obs_previous_surgeries = list(np.random.choice(self.previous_surgeries, size = total_previous_surgeries))
        self.extra_previous_surgeries = ["".join(choices(string.ascii_uppercase+string.digits, k=4)) for _ in range(n_extra_vars)]
        obs_previous_surgeries_str = ", ".join(self.obs_previous_surgeries+self.extra_previous_surgeries)
        
        # Get observed medications (not used in causal graph).
        n_meds = np.random.randint(low = 1, high = 3, size = 1).item()
        medications = ["".join(choices(string.ascii_uppercase+string.digits, k=3)) for _ in range(n_meds)]
        amounts = [str(np.random.choice([10,25,50,75,100,150],size=1).item()) for _ in range(n_meds)]
        self.medications = [x+" "+y+" mg/day" for x,y in zip(medications,amounts)]
        medications = ", ".join(self.medications)
        
        # Get observed exogenous values (binary).
        self.exog_obs = self.obs_family_history + self.obs_alleles + self.obs_previous_surgeries
        self.exog_true_binary = [int(var in self.exog_obs) for var in self.exog_names]

        # Get age and pain details.
        self.age = np.random.randint(low = 53, high = 70, size = 1).item()
        self.hours = np.random.randint(low = 2, high = 6, size = 1).item()
        if self.exog_true_binary[0] == 1: # True pain must be at least threshold.
            self.rating = np.random.randint(low = self.pain_threshold, high = 11, size = 1).item()
        else: # True pain must be below threshold.
            self.rating = np.random.randint(low = 1, high = self.pain_threshold, size = 1).item()
        self.mg = np.random.choice([75,100,250,500], size = 1).item()

        self.history = "Now, we will review the history and physical notes for patient {}.".format(self.name) 
        self.history += " History of Present Illness: {} is a {}-year-old".format(self.name, self.age)
        self.history += " {} with {} who presented to the emergency department with acute".format(self.sex, self.disease)
        self.history += " onset pain that began {} hours prior to arrival.".format(self.hours)
        self.history += " Pain was rated {}/10. The patient reports the pain has been persistent since onset.".format(self.rating)
        self.history += " The patient took aspirin ({} mg) at home with minimal relief.".format(self.mg)
        self.history += " Genetic Screening: Patient carries alleles {}.".format(obs_alleles_str)
        self.history += " Family History: {}. Medications: {}.".format(obs_family_history_str,medications)
        self.history += " Past Surgical History: Prior surgeries for {}.".format(obs_previous_surgeries_str)
        
        return self.history
        

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


    def get_factual_queries(self) -> dict:

        '''
        Returns a dictionary of all causal queries of interest mapped to their
        corresponding factual text prompts.
        '''
        
        self.f_query_dict = dict()
        outro = " Begin your response with True or False and be as concise as possible."
        for pair in [self.global_quantity]+self.local:
            effect = pair[1]
            if effect != "surgery":
                q = "Given these history and physical notes, will {} {} be {}?".format(self.var_dict.get(effect).get("endog type"),
                                               effect,
                                               self.var_dict.get(effect).get("endog magnitude"))
            else:
                q = "Given these history and physical notes, will the surgeon recommend surgery?"
            true_all = dict(zip(self.nodes,self.get_truth(intervene_node = None)))
            true_exog = dict(zip(self.exog_names,self.exog_true_binary))
            true_response = true_all.get(effect)
            self.f_query_dict[effect] = {"Prompt": q+outro, 
                                         "True endogenous": true_all,
                                         "True exogenous": true_exog,
                                         "True response": true_response}

        return self.f_query_dict


    def get_counterfactual_queries(self) -> dict:

        '''
        Returns a dictionary of all causal queries of interest mapped to their
        corresponding counterfactual text prompts (for intervention = 0 and = 1).
        '''

        if self.f_query_dict is None:
            _ = self.get_factual_queries()

        self.cf_0_query_dict = dict()
        self.cf_1_query_dict = dict()
        for pair in [self.global_quantity]+self.local:
            cause, effect = pair[0], pair[1]
            cause_type = self.var_dict.get(cause).get("endog type")
            effect_type = self.var_dict.get(effect).get("endog type")
            effect_mag = self.var_dict.get(effect).get("endog magnitude")
            cf_1 = "be " + self.var_dict.get(cause).get("endog magnitude")
            cf_0 = "not " + cf_1
            if effect == "surgery":
                outro_a = " With this new assumption, will the surgeon recommend surgery?"
            else:
                outro_a = " With this new assumption, will {} {} be {}?".format(effect_type, effect, effect_mag)
            outro_b = " Begin your response with True or False and be as concise as possible."

            # Query under counterfactual cause = True.
            if cause == "pain":
                q_1 = "Now suppose that the patient will be in significant pain regardless of all other circumstances."
            else:
                q_1 = "Now suppose that {} {} will {} regardless of all other circumstances.".format(cause_type,cause,cf_1)
            true_all = dict(zip(self.nodes,self.get_truth(intervene_node = cause, intervene_value = 1)))
            true_exog = dict(zip(self.exog_names,self.exog_true_binary))
            true_response = true_all.get(effect)
            self.cf_1_query_dict[pair] = {"Prompt": q_1 + outro_a + outro_b, 
                                          "True endogenous": true_all,
                                          "True exogenous": true_exog,
                                          "True response": true_response}

            # Query under counterfactual cause = False.
            if cause == "pain":
                q_0 = "Now suppose that the patient will not be in pain regardless of all other circumstances."
            else:
                q_0 = "Now suppose that {} {} will {} regardless of all other circumstances.".format(cause_type,cause,cf_0)
            true_all = dict(zip(self.nodes,self.get_truth(intervene_node = cause, intervene_value = 0)))
            true_response = true_all.get(effect)
            self.cf_0_query_dict[pair] = {"Prompt": q_0 + outro_a + outro_b, 
                                          "True endogenous": true_all,
                                          "True exogenous": true_exog,
                                          "True response": true_response}
            
        return self.cf_1_query_dict, self.cf_0_query_dict


class DataSetGenerator():

    '''
    Generates datasets using TaskGenerator.
    '''

    def get_dataset(self, 
                    graph_sizes: list = [[2,2,2],[3,3,3],[4,4,4],[5,5,5]],
                    n_tasks_per_size: int = 10
                    n_samples_per_task: int = 100) -> pd.DataFrame:
        
        dfs = []
        for size in graph_sizes:
            
            for task in range(n_tasks_per_size):
            
                tg = TaskGenerator(n_per_bcc = size, 
                                   bcc_types = ["cycle"]*len(size),
                                   plot = False)
                context = [tg.get_causal_context()]*n_samples_per_task
                adj_dag = [tg.adj_dag]*n_samples_per_task
                nodes_dag = [tg.nodes]*n_samples_per_task
                adj_cct = [tg.adj_cct]*n_samples_per_task
                nodes_cct = [list(tg.cct.nodes())]*n_samples_per_task
                
                global_qs = [tg.get_global()]*n_samples_per_task
                local_qs = [tg.get_local()]*n_samples_per_task
                compositions = [tg.get_compositions()]*n_samples_per_task
                
                patient_histories = []
                factual_queries = []
                cf_1_queries = []
                cf_0_queries = []
                
                for i in range(n_samples_per_task):
                    patient_histories.append(tg.get_patient_history(n_extra_vars = 2))
                    factual_queries.append(tg.get_factual_queries())
                    cf_1, cf_0 = tg.get_counterfactual_queries()
                    cf_1_queries.append(cf_1)
                    cf_0_queries.append(cf_0)
                
                df = pd.DataFrame({
                    "Context ID": [task]*n_samples_per_task, 
                    "Sample ID": range(n_samples_per_task),
                    "Nodes per BCC": size[0],
                    "Causal context": context, 
                    "DAG adjacency matrix": adj_dag, 
                    "DAG nodes": nodes_dag,
                    "CCT adjacency matrix": adj_cct, 
                    "CCT nodes": nodes_cct,
                    "Global quantity": global_qs,
                    "Local quantities": local_qs,
                    "Compositions": compositions,
                    "Patient history": patient_histories, 
                    "Factual queries": factual_queries, 
                    "Counterfactual queries (cause = True)": cf_1_queries, 
                    "Counterfactual queries (cause = False)": cf_0_queries
                })
                df.insert(0, "Task ID",
                          ['.'.join(i) for i in zip(df["Context ID"].astype(str),df["Sample ID"].astype(str),df["Nodes per BCC"].astype(str))])
                dfs.append(df)
        
        df = pd.concat(dfs).reset_index(drop = True)
        return df


    def process_prompts(self, 
                        df: pd.DataFrame) -> pd.DataFrame:

        '''
        Process dataframe returned by get_dataset(), returning factual and paired counterfactual
        prompts for easy use in benchmarking.
        '''

        dfs_fact = []
        dfs_cf = []
        
        for row in range(len(df)):
            context_id = df.loc[row, "Context ID"]
            task_id = df.loc[row, "Task ID"]
            fact = df.loc[row, "Factual queries"]
            cf_1 = df.loc[row, "Counterfactual queries (cause = True)"]
            cf_0 = df.loc[row, "Counterfactual queries (cause = False)"]
            causal_context = df.loc[row, "Causal context"]
            patient_history = df.loc[row, "Patient history"]
        
            # Get factual prompt data.
            factual_effects = []
            factual_prompts = []
            factual_true = []
            for effect,q_dict in fact.items():
                factual_effects.append(effect)
                factual_prompts.append(" ".join([causal_context,patient_history,q_dict.get("Prompt")]))
                factual_true.append(q_dict.get("True response"))
            df_fact = pd.DataFrame({"Context ID": [context_id]*len(factual_effects),
                                    "Task ID": [task_id]*len(factual_effects),
                                    "Effect": factual_effects,
                                    "Prompt": factual_prompts,
                                    "True": factual_true})
            dfs_fact.append(df_fact)
        
            # Get counterfactual prompt data.
            pairs = []
            causes = []
            effects = []
            cf_1_prompts = []
            cf_1_true = []
            cf_0_prompts = []
            cf_0_true = []
            for pair,q_dict in cf_1.items():
                pairs.append(pair)
                causes.append(pair[0])
                effects.append(pair[1])
                cf_1_prompts.append(" ".join([causal_context,patient_history,q_dict.get("Prompt")]))
                cf_1_true.append(q_dict.get("True response"))
            df_cf = pd.DataFrame({"Context ID": [context_id]*len(effects),
                                    "Task ID": [task_id]*len(effects),
                                    "Cause-effect pair": pairs,
                                    "Cause": causes,
                                    "Effect": effects,
                                    "Prompt (cause = True)": cf_1_prompts,
                                    "True (cause = True)": cf_1_true})
            for pair,q_dict in cf_0.items():
                cf_0_prompts.append(" ".join([causal_context,patient_history,q_dict.get("Prompt")]))
                cf_0_true.append(q_dict.get("True response"))
            df_cf["Prompt (cause = False)"] = cf_0_prompts
            df_cf["True (cause = False)"] = cf_0_true
            dfs_cf.append(df_cf)
        
        df_fact = pd.concat(dfs_fact).reset_index(drop = True)
        df_cf = pd.concat(dfs_cf).reset_index(drop = True)
        
        return df_fact,df_cf
            

        
        

