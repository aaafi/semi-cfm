import numpy as np
import pandas as pd
import networkx as nx
from scipy.io import loadmat
from scipy.sparse.csgraph import yen
from scipy.sparse import csr_matrix
import os
import networkx as nx
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, field

@dataclass
class LinkParameters:
    """
        A data class to store and compute fiber link parameters for optical network modeling.
    
        Example:
        --------
        >>> from CFM.core.network import LinkParameters
        
        >>> # Define link parameters
        >>> Link_params = LinkParameters()
    """
    # Fundamental constants
    c: int = 299792458
    """Speed of light"""

    #this is the reference frequency where the beta2, beta3 and beta4 are defined based on it (Taylor series expansion of
    nuu: int = 193548387096774 
    """Refrence Frequency"""
    
    # Based on some gamma measurements, it is about 0.95 which is also used in ECOC 2023 paper
    factor_gamma: float = 1.
    """Factor on the nominal gamma"""

    # Standard reference value used to model the Kerr nonlinearity in single-mode fibers
    n2: float = 2.30e-20
    """Kerr nonlinearity """

    NA: float = 0.1182 
    """numerical apperture"""

    r: float = 4.30e-6
    """Radius of the fiber"""

    # these three parameters are taken from a documentation to calculate the frequency-dependent gamma
    # gamma_per_wat_per_km_vec is for EGN model
    gamma_per_wat_per_km: float = 1.3
    """Gamma (nonlinearity parameter) of spans in 1/(Watt.Km)"""

    betta2_ps_squared_per_km: float = -21.86
    """Betta2 of each span in (picosecond^2)/(km)"""

    betta3_ps_cube_per_km: float = 0.1331
    """Beta3 (dispersion slope) of spans in ps^3/Km"""

    betta4_ps_4_per_km: float = 0
    """Beta4 of spans in ps^4/Km"""

    betta_DCU_ps_square: float = 0
    """Lumped accumulated dispersion at the end of each span in (ps)^2"""

    betta2: float = field(init=False)

    lambda_nm: float = field(init=False)

    D: float = field(init=False)


    def __post_init__(self):
        """
        Perform post-initialization computations for dependent parameters.
        """
        self.betta2 = self.betta2_ps_squared_per_km*1e-27

        self.lambda_nm = 1e9*self.c/self.nuu

        self.D = -2*np.pi*self.c*self.betta2/((self.lambda_nm*1e-9)**2)




class Link:
    """
    Class representing an optical transmission Link with its characteristics.
    """
    def __init__(self, 
                 name: str,
                 length: float,
                 num_span: int,
                 num_amp: int,
                 link_params: LinkParameters):
        """
        Initialize Band instance.

        Args:
        ---------
            name (str): 
                Link name (e.g., 'l1', 'l2')
            length (float): 
                Length of the link in km
            num_span (int):
                number of spans
            num_amp (int):
                number of amplifier in the link
            link_params (LinkParameters): 
                Fiber Link parameters

        Example:
        --------    
        >>> from sixgman.core.network import Link

        >>> # Create Link instance
        >>> link_l1 = Link(
        ... name = 'l1', # Band name
        ... length = 70, # The length of the link in km
        ... num_span = 1, # Number of spans of the link
        ... num_amp = 1,  # Number of the amplifiers
        ... link_params = LinkParameters, # the optical parameters instance
        ... )
        """

        self.name = name
        self.length = length
        self.num_span = num_span
        self.num_amp = num_amp
        self.link_params = link_params


class Topology:
    """
    Class representing the network topology.
    Creates the graph and link objects from a cost matrix.
    """

    def __init__(self,
                 netcost_matrix: np.ndarray,
                 link_params: LinkParameters,
                 span_length: float = 70):
        
        self.netcost_matrix = netcost_matrix
        self.link_params = link_params
        self.span_length = span_length

        self.graph = nx.Graph()
        self.links: Dict[Tuple[int, int], Link] = {}

        self._build_topology()


    def _build_topology(self):

        n_nodes = self.netcost_matrix.shape[0]

        edge_idx = 1

        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):

                length = self.netcost_matrix[i, j]

                if length > 0:

                    u = i + 1
                    v = j + 1

                    num_span = int(np.ceil(length / self.span_length))
                    num_amp = num_span

                    link_name = f"l{u}_{v}"

                    link = Link(
                        name=link_name,
                        length=length,
                        num_span=num_span,
                        num_amp=num_amp,
                        link_params=self.link_params
                    )

                    self.links[(u, v)] = link

                    self.graph.add_edge(
                        u,
                        v,
                        weight=length,
                        link=link,
                        idx=edge_idx
                    )

                    edge_idx += 1



    def get_link(self, u: int, v: int) -> Link:
        return self.graph[u][v]["link"]

    def get_links_list(self):
        """
        Return a list of (u, v, link) tuples for all links in the topology.
        """
        link_list = []

        for u, v, data in self.graph.edges(data=True):
            link = data["link"]
            link_list.append((u, v, link))

        return link_list
    
    def get_graph(self) -> nx.Graph:
        """
        Return the NetworkX graph of the topology.
        """
        return self.graph
