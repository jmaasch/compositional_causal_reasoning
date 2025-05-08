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
from task_generator import TaskGenerator
from dataset_generator import DataSetGenerator


class CellBio(TaskGenerator):

    '''
    Generates compositional causal reasoning tasks.

    Estimands: ATE, path-specific effects, direct effect, indirect effect
    SCM: linear causal functions, Gaussian noise
    '''


    def set_params(self):

        self.params = np.random.randint(low = 1, high = 5, size = len(self.nodes))


    def get_dag(self,
                n_per_bcc: list = [[2,2,2],[2,2,2],[2,2,2]], 
                bcc_types: list = [["cycle"]*3,["cycle"]*3,["cycle"]*3],
                label_seed: int = None,
                plot: bool = True) -> nx.classes.graph.Graph:
    
        '''
        Construct a directed acyclic graph (DAG) with exactly one root, exaclty one leaf, 
        varying numbers of biconnected components (BCCs), and varying numbers of nodes in 
        each BCC.

        Params:
            - n_per_bcc: list of lists containing number of nodes per BCC. Length of n_per_bcc
              corresponds to how many "arms" of indirect paths go from root to leaf, while the
              length of each item in n_per_bcc corresponds to the number of BCCs per "arm". If 
              n_per_bcc[0][j] == 2 for all j, this means that arm 0 is just a simple directed 
              path from root to leaf.
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

        self.subgraph_dict = dict()
        root_name = ''.join(choices(string.ascii_uppercase+string.digits, k=4))
        leaf_name = ''.join(choices(string.ascii_uppercase+string.digits, k=4))
        for j in range(len(n_per_bcc)):

            # Construct first BCC.
            if bcc_types[j][0] == "cycle":
                dag = nx.cycle_graph(n = n_per_bcc[j][0])
            elif bcc_types[j][0] == "wheel":
                dag = nx.wheel_graph(n = n_per_bcc[j][0])
        
            # Convert adjacency matrix to upper triangular to get DAG.
            adj = nx.to_numpy_array(dag)
            adj = np.triu(adj)
            dag = nx.from_numpy_array(adj) 
        
            # Get leaf.
            row_sums = adj.sum(axis = 1)
            leaf_idx = np.where(row_sums == 0)[0]

            # Add remaining BCCs.
            bccs = []
            for i in range(1,len(n_per_bcc[j])):
        
                if bcc_types[j][i] == "cycle":
                    g = nx.cycle_graph(n = n_per_bcc[j][i])
                elif bcc_types[j][i] == "wheel":
                    g = nx.wheel_graph(n = n_per_bcc[j][i])
        
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
        
            # Make acyclic, add edge weights, and add random node names.
            cyclic_dict = nx.to_dict_of_dicts(dag)
            acyclic_dict = dict()
            coeff = lambda : round(np.random.uniform(low = 0.2, high = 4.5, size = 1).item(),1)
            for parent,children in cyclic_dict.items():
                child_dict = dict()
                for child,weight in children.items():
                    if child > parent:
                        child_dict[child] = {"weight": coeff()}
                    acyclic_dict[parent] = child_dict
            dag = nx.from_dict_of_dicts(acyclic_dict, create_using = nx.DiGraph)
            if label_seed is not None:
                seed(label_seed)
            labels = [''.join(choices(string.ascii_uppercase+string.digits, k=4)) for _ in range(len(dag.nodes)-2)]
            labels = [root_name]+labels+[leaf_name]
            dag = nx.relabel_nodes(dag, dict(zip(dag.nodes,labels)))

            self.subgraph_dict[j] = dag
            
            #self.utils.plot_nx(nx.to_numpy_array(dag), 
            #                   labels = list(dag.nodes), 
            #                   figsize = (7,7), 
            #                   dpi = 50, 
            #                   node_size = 1500,
            #                   arrow_size = 20)
            
            #e_w = nx.get_edge_attributes(dag,"weight")
            #nx.draw_networkx_edge_labels(dag,
            #                             pos = nx.circular_layout(dag),
            #                             edge_labels = e_w)
            #plt.show()
            #plt.close()

        # Compose "arms" into full DAG.
        dag = nx.compose(self.subgraph_dict.get(0),self.subgraph_dict.get(1))
        if len(self.subgraph_dict.keys()) > 2:
            for k in range(2,len(self.subgraph_dict.keys())):
                dag = nx.compose(dag,self.subgraph_dict.get(k))

        # Add direct edge from root to leaf.
        dag.add_edge(root_name, leaf_name, weight = coeff())
        self.direct_effects = nx.get_edge_attributes(dag,"weight")

        if plot:
            self.utils.plot_nx(nx.to_numpy_array(dag), 
                               labels = list(dag.nodes), 
                               figsize = (7,7), 
                               dpi = 50, 
                               node_size = 1500,
                               arrow_size = 20)
            
        return dag


    def get_cct(self,
                plot: bool = True) -> nx.classes.graph.Graph:


        self.cct = dict()
        self.adj_cct = dict()
        for key,cct_sort in self.cct_sort.items():

            adj_cct = np.triu(np.ones((len(cct_sort),len(cct_sort))),k = 1)
            adj_cct = adj_cct.astype(int)
            cct = nx.from_numpy_array(adj_cct, create_using = nx.DiGraph)
            cct = nx.relabel_nodes(cct, dict(zip(cct.nodes,cct_sort)))
            self.adj_cct[key] = adj_cct
            self.cct[key] = cct
    
            if plot:
                self.utils.plot_nx(nx.to_numpy_array(self.cct.get(key)), 
                                   labels = list(self.cct.get(key).nodes), 
                                   figsize = (7,7), 
                                   dpi = 50, 
                                   node_size = 1500,
                                   arrow_size = 20)
    
        return self.cct


    def get_cct_all_paths(self) -> list:

        '''
        Getter for composition cause-effect pairs for inductive CCR evaluation
        using Algorithm 1 / Theorem 1.

        Input is commutative cut tree (CCT), not the original causal DAG.
        '''

        self.path_dict = dict()
        for key,subgraph in self.cct.items():
            self.path_dict[key] = list(nx.all_simple_paths(subgraph, self.root, self.leaf))

        return self.path_dict


    def get_cutpoints(self, 
                      dag: nx.classes.graph.Graph, 
                      topological_sort: bool = True) -> list:
        
        '''
        Getter for a topological sort of cutpoints.
        '''

        self.cutpoint_dict = dict()
        for key,subgraph in self.subgraph_dict.items():
            self.cutpoint_dict[key] = self._get_cutpoints(dag = subgraph, 
                                                          topological_sort = topological_sort)

        return self.cutpoint_dict


    def get_cct_sort(self):

        self.cct_sort = dict()
        for key,nodes in self.cutpoint_dict.items():
            self.cct_sort[key] = [self.root] + nodes + [self.leaf]

        return self.cct_sort


    def get_cause_effect_pairs(self,
                               dag: nx.classes.graph.Graph) -> list:

        '''
        Getter for all relevant cause-effect pairs for inductive CCR evaluation
        using Algorithm 1 / Theorem 1.
        '''

        combos = list(itertools.combinations(nx.topological_sort(dag),2))
        
        # This step appears redundant, but I will keep this just in case 
        # the apparent sorting by itertools is not consistent.
        cause_effect_pairs = [x for x in combos if list(dag.nodes).index(x[0]) < list(dag.nodes).index(x[1])]
        return cause_effect_pairs


    def get_local(self) -> list:

        '''
        Getter for local quantity cause-effect pairs for inductive CCR evaluation
        using Algorithm 1 / Theorem 1.

        Returns a dictionary mapping subgraphs (each with their own CCT) to local quantities.
        '''

        self.local_dict = dict()
        for key,cct in self.cct.items():
            all_pairs = self.get_cause_effect_pairs(cct)
            self.local_dict[key] = [x for x in all_pairs if x != (self.root, self.leaf)]

        return self.local_dict
        

    def get_compositions(self) -> list:

        '''
        Getter for composition cause-effect pairs for inductive CCR evaluation
        using Algorithm 1 / Theorem 1.
        '''

        self.get_cct_all_paths()
        self.comp_dict = dict()
        for key,subgraph in self.cct.items():
            compositions = []
            paths = self.path_dict.get(key)
            for path in paths:
                comp = [(path[i],path[i+1]) for i in range(len(path)-1)] 
                if len(comp) >= 2:
                    compositions.append(comp)
            self.comp_dict[key] = compositions

        return self.comp_dict


    def get_causal_context(self) -> str:

        '''
        Define causal model in text.
        '''

        # Set coefficients for structural equations.
        self.set_params()
    
        # Get variable metadata for context prompt.
        self.var_dict = dict()
        self.cell_type = ''.join(choices(string.ascii_uppercase+string.digits, k=6))
        self.organism = ''.join(choices(string.ascii_uppercase+string.digits, k=6))
        self.compound = ''.join(choices(string.ascii_uppercase+string.digits, k=6))
        #exog = ["enzyme"]*len(self.nodes)
        #endog = ["mRNA"]*len(self.nodes)
        #units = "pg" #(picograms)
        #units = "g/mL"

        for var,u in zip(self.nodes,self.exog_names):
            parents = self.get_parents(var, return_idx = False)
            self.var_dict[var] = {"parents": parents,
                                  "endog type": "mRNA", 
                                  "exog var name": u, 
                                  "exog type": "enzyme"}

        '''
        A cellular biologist is studying the impacts of exposure to compound {} on  
        transcription and translation in cell type {} of organism {}. When stilumated with compound {}, 
        cell type {} will produce mRNA transcripts for gene {} at {} times 
        the current volume of enzyme {}. The cell will produce mRNA transcripts for gene {} at {} times the 
        current volume of enzyme {} plus {} times the current volume of {} transcripts. The cell will produce 
        mRNA transcripts for gene {} at {} times the current volume of enzyme {} plus {} times the current 
        volume of {} transcripts plus {} times the current volume of {} transcripts. The total volume of protein
        {} will be three times the current volume of {} transcripts plus the current volume of enzyme {}. 

        Do this for all paths from root to leaf.
        
        Assume that all factorus influencing the transcription and translation of these
        macromolecules are described here.

        At the time of the experiment, the biologist measures {} pg of enzyme {}, {} pg enzyme {}, etc... How 
        much protein {} will be present in the cell?

        Now suppose that the biologist can artificially induce the cell to produce more/less of a 
        specific transcript, and now X pg of transcript J are present in the cell regardless of all 
        other circumstances. With this new assumption, how much protein will be present in the cell?
        '''

        # Construct prompt.
        intro = "A cellular biologist is studying the impacts of exposure to compound {}".format(self.compound)
        intro += " on transcription and translation in cell type {} of organism {}.".format(self.cell_type,self.organism)
        intro += " When stilumated with compound {}, cell type {}".format(self.compound,self.cell_type)
        outro = "Assume that all factors influencing the transcription and translation of"
        outro += " these molecules are fully described here."
        strngs = [intro]
        for var,terms in self.var_dict.items():
            parents = terms.get("parents")
            exog = "the patient "+terms.get("exog type")+" "+terms.get("exog var name")
            if len(parents) > 0:
                strng = " When stimulated with compound {}, cell type {}".format(self.compound,self.cell_type)
                strng += " will produce mRNA transcripts for gene {}".format(var)
                strng += " equal in volume to the current volume of enzyme {}.".format(terms.get("exog var name"))
            else:
                parent_strngs = []
                strng = "The cell will produce mRNA transcripts for gene {}".format(var)
                for parent in parents:
                    parent_strngs.append(" at {} times the current volume of {} transcripts".format(self.direct_effects.get((parent,var)),
                                                                                                    parent))
                parent_strngs.append("the current volume of enzyme {}".format(terms.get("exog var name")))
                parent_strngs = " plus ".join(parent_strngs)+"."
            strng += parent_strngs
            strngs.append(strng)
        strngs.append(outro)
        
        self.causal_context = " ".join(strngs)
        return self.causal_context


    def get_sample_context(self,
                           n_extra_vars: int = 5) -> str:

        '''
        Sample exogenous variables and construct text prompt.
        '''

        # Get extraneous details.
        self.extra_names = [''.join(choices(string.ascii_uppercase+string.digits, k=4)) for _ in range(n_extra_vars)]

        # Get observed quantities.
        self.exog_obs = [np.random.uniform(low = 0.1, high = 3.0, size = 1).item() for _ in range(len(self.nodes))]
        self.extra_obs = [np.random.uniform(low = 0.1, high = 3.0, size = 1).item() for _ in range(len(self.nodes))]

        # Construct context.
        self.sample_context = "At the time of the experiment, the biologist measures"
        volumes = [str(pg)+" pg of enzyme "+u for pg,u in zip(self.exog_obs,self.exog_names)]
        volumes += [str(pg)+" pg of enzyme "+u for pg,u in zip(self.extra_obs,self.extra_names)]
        shuffle(volumes)
        self.sample_context += ", ".join(volumes[:-1])
        self.sample_context += " and "+volumes[-1]+"."

        return self.sample_context


    def get_factual_queries(self) -> dict:

        '''
        Returns a dictionary of all causal queries of interest mapped to their
        corresponding factual text prompts.
        '''
        
        self.f_query_dict = dict()
        outro = " Begin your response with Yes or No and be as concise as possible."
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
            outro_b = " Begin your response with Yes or No and be as concise as possible."

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


