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


class DataSetGenerator():


    def __init__(self):
        
        # For utility functions.
        self.utils = Utils()

        
    def get_dataset(self, 
                    task_generator, # ClinicalNotes, CandyParty, etc.
                    graph_sizes: list = [[2,2,2],[3,3,3],[4,4,4]],
                    n_tasks_per_size: int = 10,
                    n_samples_per_task: int = 1000,
                    reps_per_sample: int = None,
                    n_extra_vars: int = None) -> pd.DataFrame:
        
        dfs = []
        for size in graph_sizes:
            
            start = graph_sizes.index(size)*n_tasks_per_size
            for task in range(start,start+n_tasks_per_size):

                # Init task generator.
                tg = task_generator(n_per_bcc = size, 
                                    bcc_types = ["cycle"]*len(size),
                                    plot = False)

                # Get metadata.
                context = [tg.get_causal_context()]*n_samples_per_task
                adj_dag = [tg.adj_dag]*n_samples_per_task
                nodes_dag = [tg.nodes]*n_samples_per_task
                adj_cct = [tg.adj_cct]*n_samples_per_task
                nodes_cct = [list(tg.cct.nodes())]*n_samples_per_task
                exog_names = [tg.exog_names]*n_samples_per_task
                p = [tg.p]*n_samples_per_task
                
                global_qs = [tg.get_global()]*n_samples_per_task
                local_qs = [tg.get_local()]*n_samples_per_task
                compositions = [tg.get_compositions()]*n_samples_per_task
                
                sample_contexts = []
                factual_queries = []
                cf_1_queries = []
                cf_0_queries = []
                
                for i in range(n_samples_per_task):
                    if n_extra_vars is not None:
                        sample_contexts.append(tg.get_sample_context(n_extra_vars = n_extra_vars))
                    else:
                        sample_contexts.append(tg.get_sample_context())
                    factual_queries.append(tg.get_factual_queries())
                    cf_1, cf_0 = tg.get_counterfactual_queries()
                    cf_1_queries.append(cf_1)
                    cf_0_queries.append(cf_0)
                
                df = pd.DataFrame({
                    "Context ID": task, 
                    "Sample ID": range(n_samples_per_task),
                    "Nodes per BCC": [size]*n_samples_per_task,
                    "DAG adjacency matrix": adj_dag, 
                    "DAG nodes": nodes_dag,
                    "CCT adjacency matrix": adj_cct, 
                    "CCT nodes": nodes_cct,
                    "Exogenous variables": exog_names,
                    "Bernoulli parameters": p,
                    "Global quantity": global_qs,
                    "Local quantities": local_qs,
                    "Compositions": compositions,
                    "Causal context": context, 
                    "Sample context": sample_contexts, 
                    "Factual queries": factual_queries, 
                    "Counterfactual queries (cause = True)": cf_1_queries, 
                    "Counterfactual queries (cause = False)": cf_0_queries
                })
                df.insert(0, "Task ID",
                          ['.'.join(i) for i in zip(df["Context ID"].astype(str),df["Sample ID"].astype(str))])
                dfs.append(df)
        
        self.df = pd.concat(dfs).reset_index(drop = True)

        # Replicate samples if desired.
        if reps_per_sample is not None:
            rep_ids = list(np.arange(reps_per_sample))*len(self.df)
            self.df = pd.DataFrame(np.repeat(self.df.values, repeats = reps_per_sample, axis = 0), 
                                   columns = self.df.columns)
            self.df.insert(3, "Replicate ID", rep_ids)

        return self.df


    def process_prompts(self) -> pd.DataFrame:

        '''
        Process dataframe returned by get_dataset(), returning factual and paired counterfactual
        prompts for easy use in benchmarking.
        '''

        dfs_fact = []
        dfs_cf = []
        
        for row in range(len(self.df)):
            context_id = self.df.loc[row, "Context ID"]
            task_id = self.df.loc[row, "Task ID"]
            sample_id = self.df.loc[row, "Sample ID"]
            if "Replicate ID" in self.df.columns:
                rep_id = self.df.loc[row, "Replicate ID"]
            n_bcc = self.df.loc[row, "Nodes per BCC"]
            fact = self.df.loc[row, "Factual queries"]
            cf_1 = self.df.loc[row, "Counterfactual queries (cause = True)"]
            cf_0 = self.df.loc[row, "Counterfactual queries (cause = False)"]
            causal_context = self.df.loc[row, "Causal context"]
            sample_context = self.df.loc[row, "Sample context"]
        
            # Get factual prompt data.
            factual_effects = []
            factual_prompts = []
            factual_true = []
            for effect,q_dict in fact.items():
                factual_effects.append(effect)
                factual_prompts.append(" ".join([causal_context,sample_context,q_dict.get("Prompt")]))
                factual_true.append(q_dict.get("True response"))
            df_fact = pd.DataFrame({"Task ID": task_id,
                                    "Context ID": context_id,
                                    "Sample ID": sample_id,
                                    "Nodes per BCC": [n_bcc]*len(factual_effects),
                                    "Effect": factual_effects,
                                    "Prompt": factual_prompts,
                                    "True": factual_true})
            if "Replicate ID" in self.df.columns:
                df_fact.insert(3, "Replicate ID", rep_id)
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
                cf_1_prompts.append(" ".join([causal_context,sample_context,q_dict.get("Prompt")]))
                cf_1_true.append(q_dict.get("True response"))
            df_cf = pd.DataFrame({"Task ID": task_id,
                                  "Context ID": context_id,
                                  "Sample ID": sample_id,
                                  "Nodes per BCC": [n_bcc]*len(causes),
                                  "Cause-effect pair": pairs,
                                  "Cause": causes,
                                  "Effect": effects,
                                  "Prompt (cause = True)": cf_1_prompts,
                                  "True (cause = True)": cf_1_true})
            for pair,q_dict in cf_0.items():
                cf_0_prompts.append(" ".join([causal_context,sample_context,q_dict.get("Prompt")]))
                cf_0_true.append(q_dict.get("True response"))
            df_cf["Prompt (cause = False)"] = cf_0_prompts
            df_cf["True (cause = False)"] = cf_0_true
            if "Replicate ID" in self.df.columns:
                df_cf.insert(3, "Replicate ID", rep_id)
            dfs_cf.append(df_cf)
        
        self.df_fact = pd.concat(dfs_fact).reset_index(drop = True)
        self.df_cf = pd.concat(dfs_cf).reset_index(drop = True)
        
        return self.df_fact,self.df_cf
        

    def get_pns_ate(self,
                    df: pd.DataFrame, 
                    verbose: bool = True,
                    return_value: str = "pns") -> float:
        
        pns = self.utils.get_pns_direct(df, 
                                        y_do_x1 = "True (cause = True)", 
                                        y_do_x0 = "True (cause = False)")
        ate = self.utils.get_ate(df,
                                 y_do_x1 = "True (cause = True)", 
                                 y_do_x0 = "True (cause = False)")
        if verbose:
            print("-- PNS = {} | ATE = {} --".format(pns,ate))
            
        if return_value == "pns":
            return pns
        elif return_value == "ate":
            return ate
        else:
            return pns,ate


    def get_pns_dict(self, 
                     verbose: bool = False) -> dict:

        '''
        Get dictionary mapping cause-effect pairs to their PNS value.
        '''

        self.pns_dict = dict()
        for context_id in self.df_cf["Context ID"].unique():
            df_context = self.df_cf[self.df_cf["Context ID"] == context_id]
            pair_dict = dict()
            
            # Get local and global PNS.
            for pair in df_context["Cause-effect pair"].unique():
                pair_dict[str(pair)] = self.get_pns_ate(df_context[df_context["Cause-effect pair"] == pair], 
                                                        verbose = verbose,
                                                        return_value = "pns")

            # # Get PNS for compositions.
            df_comp = self.df[self.df["Context ID"] == context_id]
            compositions = df_comp["Compositions"].value_counts().index.item()
            for comp in compositions:
                pns = 1
                for pair in comp:
                    pns *= pair_dict.get(str(pair))
                pair_dict[str(comp)] = pns
            self.pns_dict[context_id] = pair_dict
        
        return self.pns_dict


    def get_internal_consistency_thresholds(self, 
                                            multiplier: float = 2.0) -> dict:
        
        '''
        Return a dictionary that maps compositions to their correctness threshold
        for internal compositional consistency evaluation. Thresholds are the RAE
        for each composition relative to the global quantity of interest, times a
        multiplier of the user's choice. 

        RAE = [abs(global PNS - composition PNS) / global PNS]
        Threhold = RAE*multiplier
        
        This method of obtaining the threshold accounts for the innate error owed
        to PNS estimation on finite samples, while the multiplier represents the
        user's tolerance level for errors larger than the finite sample error.
        '''

        self.threshold_dict = dict()
        for context in self.df["Context ID"].unique():
            context_dict = dict()
            df_context = self.df[self.df["Context ID"] == context]
            glo = df_context["Global quantity"].unique()[0]
            compositions = df_context["Compositions"].value_counts().index.item()
            for comp in compositions:
                glo_pns = self.pns_dict.get(context).get(str(glo))
                comp_pns = self.pns_dict.get(context).get(str(comp))
                context_dict[str(comp)] = (abs(glo_pns - comp_pns) / glo_pns)*multiplier
            self.threshold_dict[context] = context_dict
        
        return self.threshold_dict



        