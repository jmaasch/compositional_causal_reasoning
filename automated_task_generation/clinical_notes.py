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


class ClinicalNotes(TaskGenerator):

    '''
    Generates compositional causal reasoning tasks.
    '''


    def get_causal_functions(self):

        return ["or"]*(len(self.nodes)-1)+["and"]


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


    def get_causal_context(self) -> str:

        '''
        Define causal model in text.
        '''
    
        # Get variable metadata for context prompt.
        self.var_dict = dict()
        self.alleles = []             # exogenous variables.
        self.fam_hist = []      # exogenous variables.
        self.prev_surg = []  # exogenous variables.
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
                self.fam_hist.append(u)
            elif "surgery" in exog:
                self.prev_surg.append(u)
        
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
        intro += " depending on genetics, Sample context, vital signs, and lab results. The patient will experience" 
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


    def get_sample_context(self,
                           n_extra_vars: int = 2) -> str:

        '''
        Sample exogenous variables and construct text prompt.
        '''

        # Get patient sex and name according to sex.
        self.sex = np.random.choice(["male", "female"], size = 1).item()
        f = Faker()
        if self.sex == "female":
            self.name = f.name_female()
        else:
            self.name = f.name_male()


        # Get observed exogenous variables based on user-selected Bernoulli parameters.
        # These are the same parameters used to sample exogenous variables in self.sample_scm().
        bern = lambda p: np.random.binomial(n = 1, p = p, size = 1).item()
        self.exog_true_binary = [bern(p) for p,_ in zip(self.p,self.exog_names)]
        self.exog_obs = [x for x,y in zip(self.exog_names,self.exog_true_binary) if y == 1]
        
        # Get observed alleles.
        self.alleles_obs = [x for x in self.exog_obs if x in self.alleles]
        self.alleles_extra = ["".join(choices(string.ascii_uppercase+string.digits, k=4)) for _ in range(n_extra_vars)]
        alleles_obs_str = ", ".join(self.alleles_obs+self.alleles_extra)
        
        # Get observed family medical history.
        self.fam_hist_obs = [x for x in self.exog_obs if x in self.fam_hist]
        self.fam_hist_extra = ["".join(choices(string.ascii_uppercase+string.digits, k=4)) for _ in range(n_extra_vars)]
        fam_hist_obs_str = ", ".join(self.fam_hist_obs+self.fam_hist_extra)
        
        # Get observed surgical history.
        self.prev_surg_obs = [x for x in self.exog_obs if x in self.prev_surg]
        self.prev_surg_extra = ["".join(choices(string.ascii_uppercase+string.digits, k=4)) for _ in range(n_extra_vars)]
        prev_surg_obs_str = ", ".join(self.prev_surg_obs+self.prev_surg_extra)
        
        # Get observed medications (not used in causal graph).
        n_meds = np.random.randint(low = 1, high = 3, size = 1).item()
        medications = ["".join(choices(string.ascii_uppercase+string.digits, k=3)) for _ in range(n_meds)]
        amounts = [str(np.random.choice([10,25,50,75,100,150],size=1).item()) for _ in range(n_meds)]
        self.medications = [x+" "+y+" mg/day" for x,y in zip(medications,amounts)]
        medications = ", ".join(self.medications)

        # Get age and pain details.
        self.age = np.random.randint(low = 53, high = 70, size = 1).item()
        self.hours = np.random.randint(low = 2, high = 6, size = 1).item()
        if self.exog_true_binary[0] == 1: # True pain must be at least threshold.
            self.rating = np.random.randint(low = self.pain_threshold, high = 11, size = 1).item()
        else: # True pain must be below threshold.
            self.rating = np.random.randint(low = 3, high = self.pain_threshold, size = 1).item()
        self.mg = np.random.choice([75,100,250,500], size = 1).item()

        self.history = "Now, we will review the history and physical notes for patient {}.".format(self.name) 
        self.history += " History of Present Illness: {} is a {}-year-old".format(self.name, self.age)
        self.history += " {} with {} who presented to the emergency department with acute".format(self.sex, self.disease)
        self.history += " onset pain that began {} hours prior to arrival.".format(self.hours)
        self.history += " Pain was rated {}/10. The patient reports the pain has been persistent since onset.".format(self.rating)
        self.history += " The patient took aspirin ({} mg) at home with minimal relief.".format(self.mg)
        self.history += " Genetic Screening: Patient carries alleles {}.".format(alleles_obs_str)
        self.history += " Family History: {}. Medications: {}.".format(fam_hist_obs_str,medications)
        self.history += " Past Surgical History: Prior surgeries for {}.".format(prev_surg_obs_str)
        
        return self.history


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


