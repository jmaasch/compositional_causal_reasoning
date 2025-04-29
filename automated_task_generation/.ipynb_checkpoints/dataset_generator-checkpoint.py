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
                    graph_sizes: list = [[2,2,2],[3,3,3],[4,4,4],[5,5,5]],
                    n_tasks_per_size: int = 10,
                    n_samples_per_task: int = 100,
                    n_extra_vars: int = 2) -> pd.DataFrame:

        self.df = pd.DataFrame({"Context ID": [],
                                "Global quantity": [], 
                                "Compositions": []})
        return self.df


    def process_prompts(self) -> pd.DataFrame:

        '''
        Process dataframe returned by get_dataset(), returning factual and paired counterfactual
        prompts for easy use in benchmarking.
        '''

        self.df_cf = pd.DataFrame({"Context ID": [],
                                   "Cause-effect pair": []})
        return self.df_cf
            

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
        multiplier of the users choice. 

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



        