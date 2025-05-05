# General importations.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class Utils():
    
    def get_cct(self,
                n_cutpoints: int = 1,
                names: list = None,
                plot: bool = True) -> nx.classes.graph.Graph:

        '''
        Generates the commutative cut tree associated with a given number of cutpoints.
        Optionally, the user can input node names.
        '''
    
        adj_cct = np.triu(np.ones((n_cutpoints+2,n_cutpoints+2)), k = 1)
        cct = nx.from_numpy_array(adj_cct, create_using = nx.DiGraph)
        if names is not None:
            cct = nx.relabel_nodes(cct, dict(zip(cct.nodes,names)))
        return cct

    
    def get_total_paths_cct(self, n: int) -> int:
    
        '''
        n = total nodes in CCT
        '''
    
        return 2**(n-2)

    
    def get_prc_direct(self,
                       df: pd.DataFrame, 
                       x: str, 
                       y: str, 
                       y_do_x0: str, 
                       y_do_x1: str) -> dict:

        '''
        Compute the probabilities of causation directly from observed and interventional data.
        '''
    
        res = dict()
        df = df.astype("bool")
        res['PN'] = np.mean(~df[df[x] & df[y]][y_do_x0])
        res['PS'] = np.mean(df[~df[x] & ~df[y]][y_do_x1])
        res['PNS'] = np.mean(df[y_do_x1] & ~df[y_do_x0])
        
        return res


    def get_pns_direct(self,
                       df: pd.DataFrame, 
                       y_do_x0: str, 
                       y_do_x1: str) -> float:

        '''
        Compute the PNS directly from interventional data.
        '''
    
        df = df.astype("bool")
        return np.mean(df[y_do_x1] & ~df[y_do_x0])


    def get_ate(self,
                df: pd.DataFrame, 
                y_do_x1: str,
                y_do_x0: str) -> float:
        
        return df[y_do_x1].mean() - df[y_do_x0].mean()


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


    def get_rae(self,
                true: np.array, 
                pred: np.array) -> float:

        return abs(true - pred) / true



    