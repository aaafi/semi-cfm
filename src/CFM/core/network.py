import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
from dataclasses import dataclass, field


@dataclass
class LinkParameters:
    """
    Data class storing physical and dispersion parameters for optical fiber links.

    Attributes:
    -----------
        c (int): Speed of light in vacuum in m/s (default: 299,792,458 m/s).
        nuu (int): Reference optical frequency for dispersion Taylor expansion in Hz.
        factor_gamma (float): Scaling factor applied to nominal Kerr nonlinearity.
        n2 (float): Nonlinear refractive index coefficient in m^2/W.
        NA (float): Numerical aperture of the fiber core.
        r (float): Core radius of the fiber in meters.
        gamma_per_wat_per_km (float): Nonlinear parameter gamma in 1/(W*km).
        betta2_ps_squared_per_km (float): Group velocity dispersion parameter beta_2 in ps^2/km.
        betta3_ps_cube_per_km (float): Third-order dispersion parameter beta_3 in ps^3/km.
        betta4_ps_4_per_km (float): Fourth-order dispersion parameter beta_4 in ps^4/km.
        betta_DCU_ps_square (float): Lumped dispersion compensation module value in ps^2.
        betta2 (float): Calculated second-order dispersion coefficient in s^2/m.
        lambda_nm (float): Reference carrier wavelength in nanometers.
        D (float): Chromatic dispersion parameter in s/(m^2).

    Example:
    --------
    >>> from CFM.core.network import LinkParameters
    >>> link_params = LinkParameters()
    """

    c: int = 299792458
    nuu: int = 193548387096774
    factor_gamma: float = 1.0
    n2: float = 2.30e-20
    NA: float = 0.1182
    r: float = 4.30e-6
    gamma_per_wat_per_km: float = 1.3
    betta2_ps_squared_per_km: float = -21.86
    betta3_ps_cube_per_km: float = 0.1331
    betta4_ps_4_per_km: float = 0.0
    betta_DCU_ps_square: float = 0.0

    betta2: float = field(init=False)
    lambda_nm: float = field(init=False)
    D: float = field(init=False)

    def __post_init__(self):
        """
        Compute derived dispersion coefficients and operational wavelength.
        """
        self.betta2 = self.betta2_ps_squared_per_km * 1e-27
        self.lambda_nm = 1e9 * self.c / self.nuu
        self.D = -2 * np.pi * self.c * self.betta2 / ((self.lambda_nm * 1e-9) ** 2)


class Link:
    """
    Representation of an amplified optical fiber link spanning network nodes.

    Attributes:
    -----------
        name (str): Unique string identifier for the fiber link.
        length (float | list[float]): Link physical length or list of span lengths in km.
        num_span (int): Total number of fiber spans comprising the link.
        num_amp (int): Total number of optical inline amplifiers deployed on the link.
        link_params (LinkParameters): Physical and nonlinear parameters of the fiber.
    """

    def __init__(self, 
                 name: str,
                 length: float,
                 num_span: int,
                 num_amp: int,
                 link_params: LinkParameters):
        """
        Initialize an optical transmission Link instance.

        Args:
        ---------
            name (str): 
                Link identifier (e.g., 'l1', 'l1_2').
            length (float | list[float]): 
                Total length of the link or list of span lengths in km.
            num_span (int): 
                Number of spans across the link.
            num_amp (int): 
                Number of inline optical amplifiers along the link.
            link_params (LinkParameters): 
                Physical parameter specifications for the fiber spans.

        Example:
        ---------
        >>> from CFM.core.network import Link, LinkParameters
        >>> params = LinkParameters()
        >>> link_l1 = Link(
        ...     name='l1',
        ...     length=70,
        ...     num_span=1,
        ...     num_amp=1,
        ...     link_params=params
        ... )
        """
        self.name = name
        self.length = length
        self.num_span = num_span
        self.num_amp = num_amp
        self.link_params = link_params


class Topology:
    """
    Graph representation of the optical physical network topology.

    Constructs a NetworkX graph containing weighted fiber spans, routing attributes, 
    and modular Link objects extracted from an adjacency cost matrix.

    Attributes:
    -----------
        netcost_matrix (np.ndarray): Symmetric adjacency matrix where entries represent lengths in km.
        link_params (LinkParameters): Fiber characteristics applied across all constructed links.
        span_length (float): Standard maximum span length used for span discretization in km.
        graph (nx.Graph): NetworkX graph object containing vertices, edges, and Link instances.
        links (Dict[Tuple[int, int], Link]): Hash map mapping node pairs (u, v) to Link instances.
    """

    def __init__(self,
                 netcost_matrix: np.ndarray,
                 link_params: LinkParameters,
                 span_length: float = 70):
        """
        Initialize the network topology from an adjacency cost matrix.

        Args:
        ---------
            netcost_matrix (np.ndarray): 
                2D square matrix where entry (i, j) indicates the link length in km.
            link_params (LinkParameters): 
                Physical fiber link parameters applied to all spans.
            span_length (float, optional): 
                Target span discretization length in km. Default is 70 km.

        Example:
        ---------
        >>> from CFM.core.network import Topology, LinkParameters
        >>> import numpy as np
        >>> cost_mat = np.array([[0, 150], [150, 0]])
        >>> params = LinkParameters()
        >>> topo = Topology(netcost_matrix=cost_mat, link_params=params, span_length=75)
        """
        self.netcost_matrix = netcost_matrix
        self.link_params = link_params
        self.span_length = span_length

        self.graph = nx.Graph()
        self.links: Dict[Tuple[int, int], Link] = {}

        self._build_topology()

    def _build_topology(self):
        """
        Assemble the graph nodes, edges, and Link configurations from the cost matrix.
        """
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
        """
        Retrieve the Link instance corresponding to an edge between nodes u and v.

        Args:
        ---------
            u (int): Source node index (1-based).
            v (int): Destination node index (1-based).

        Returns:
        ---------
            Link: Link object assigned to the queried edge.
        """
        return self.graph[u][v]["link"]

    def get_links_list(self) -> List[Tuple[int, int, Link]]:
        """
        Return all links configured within the topology as (source, destination, link) tuples.

        Returns:
        ---------
            List[Tuple[int, int, Link]]: List of tuples containing endpoints and the Link instance.
        """
        link_list = []

        for u, v, data in self.graph.edges(data=True):
            link = data["link"]
            link_list.append((u, v, link))

        return link_list

    def get_graph(self) -> nx.Graph:
        """
        Retrieve the underlying NetworkX graph representation of the network.

        Returns:
        ---------
            nx.Graph: NetworkX graph object.
        """
        return self.graph