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


class FluVaccine(TaskGenerator):

    '''
    Generates compositional causal reasoning tasks.
    '''

    def set_thresholds(self):

        '''
        Set thresholds for happiness.
        '''
        
        self.thresh = [int(x*10) for x in self.p]

        
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

        self.set_thresholds()
        intro = "A group of friends is considering whether or not to get vaccinated against the flu this year. "
        self.clauses = [" will get vaccinated if she had a high fever or bronchitis for at least ", # clauses[0]
                        " days in the previous flu season",                   # clauses[1]
                        " gets vaccinated",                                   # clauses[2]
                        " During the previous flu season, ",                  # clauses[3]
                        " had a high fever or major cough for ",              # clauses[4]
                        " days"]                                              # clauses[5]

        strings = [intro]
        for i in range(len(self.nodes)):
            #parents = list(dag.predecessors(node)) # Auto-generated adjdagacency matrix is not upper triangular.
            #parents = dag.pred[node]
            parents_idx = np.nonzero(self.adj_dag[:,i])[0]
            parents = [self.nodes[j] for j in parents_idx]
            if len(parents) == 0:
                string = self.nodes[i] + self.clauses[0] + str(self.thresh[i]) + self.clauses[1] + ". "
            else:
                string = self.nodes[i] + self.clauses[0] + str(self.thresh[i]) + self.clauses[1]
                for parent in parents:
                    if self.causal_functions[i] == "or":
                        string += " or if " + parent + self.clauses[2]
                    else: 
                        string += " and " + parent + self.clauses[2]
                string += ". "
            strings.append(string)
        self.causal_context = "".join(strings)

        return self.causal_context


    def get_sample_context(self) -> str:

        '''
        Sample exogenous variables and construct text prompt.
        '''

        # Get observed exogenous variables based on user-selected Bernoulli parameters.
        # These are the same parameters used to sample exogenous variables in self.sample_scm().
        bern = lambda p : np.random.binomial(n = 1, p = p, size = 1).item()
        randint = lambda lo_hi : np.random.randint(low = lo_hi[0], high = lo_hi[1], size = 1).item()
        exog_types = ["a high fever", "bronchitis", "a low-grade fever",  
                      "a cough", "a runny nose", "stomach pain"]
        self.exog_true_binary = [bern(p) for p,_ in zip(self.p,self.exog_names)]
        self.exog_obs = []
        self.exog_obs_alt = []
        for i in range(len(self.exog_true_binary)):
            if self.exog_true_binary[i] == 1:
                self.exog_obs.append(randint([self.thresh[i],self.thresh[i]+3]))
                self.exog_obs_alt.append(randint([self.thresh[i],self.thresh[i]+3]))
            else:
                self.exog_obs.append(randint([2,self.thresh[i]]))
                self.exog_obs_alt.append(randint([2,self.thresh[i]]))

        self.sample_context = self.clauses[3][:]
        for i in range(len(self.nodes)):
            self.sample_context += self.nodes[i]+" had "
            exog_strs = []
            days = [self.exog_obs[i], self.exog_obs_alt[i]] + [randint([1,10]) for _ in range(len(exog_types)-2)]
            for j in range(len(exog_types)):
                exog_strs.append(exog_types[j]+" for {} days".format(days[j]))
            shuffle(exog_strs)
            self.sample_context += ", ".join(exog_strs[:-1])
            self.sample_context += ", and "+exog_strs[-1]+". "

        return self.sample_context


    def get_factual_queries(self) -> dict:

        '''
        Returns a dictionary of all causal queries of interest mapped to their
        corresponding factual text prompts.
        '''
        
        self.f_query_dict = dict()
        for pair in [self.global_quantity]+self.local:
            effect = pair[1]
            q = "Did {} get vaccinated?".format(effect)
            q += " Begin your response with Yes or No and be as concise as possible."
            true_all = dict(zip(self.nodes,self.get_truth(intervene_node = None)))
            true_exog = dict(zip(self.exog_names,self.exog_true_binary))
            true_response = true_all.get(effect)
            self.f_query_dict[effect] = {"Prompt": q, 
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

            # Extract cause and effect nodes.
            cause, effect = pair[0], pair[1]

            # Query under counterfactual cause = True.
            q_1 = "Now, suppose that {} got vaccinated regardless of all other circumstances.".format(cause)
            q_1 += " With this new assumption, did {} get vaccinated?".format(effect)
            q_1 += " Begin your response with Yes or No and be as concise as possible."
            true_all = dict(zip(self.nodes,self.get_truth(intervene_node = cause, intervene_value = 1)))
            true_exog = dict(zip(self.exog_names,self.exog_true_binary))
            true_response = true_all.get(effect)
            self.cf_1_query_dict[pair] = {"Prompt": q_1, 
                                          "True endogenous": true_all,
                                          "True exogenous": true_exog,
                                          "True response": true_response}

            # Query under counterfactual cause = False.
            q_0 = "Now, suppose that {} did not get vaccinated regardless of all other circumstances.".format(cause)
            q_0 += " With this new assumption, did {} get vaccinated?".format(effect)
            q_0 += " Begin your response with Yes or No and be as concise as possible."
            true_all = dict(zip(self.nodes,self.get_truth(intervene_node = cause, intervene_value = 0)))
            true_exog = dict(zip(self.exog_names,self.exog_true_binary))
            true_response = true_all.get(effect)
            self.cf_0_query_dict[pair] = {"Prompt": q_0, 
                                          "True endogenous": true_all,
                                          "True exogenous": true_exog,
                                          "True response": true_response}
            
        return self.cf_1_query_dict, self.cf_0_query_dict


        