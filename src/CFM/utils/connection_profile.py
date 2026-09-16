from typing import List, Tuple, Any
import numpy as np
import networkx as nx


def build_connection_profile(G: nx.Graph,
                             core_nodes: List[Any],
                             kSP: int = 3,
                             Lspan: float = 80.0) -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    """
    Build connection profiles and routing paths between core nodes across a network graph[cite: 10].

    Computes node degrees, updates edge attributes with span discretization counts[cite: 10],
    and evaluates k-shortest paths (kSP) for all unique pairs of core nodes[cite: 10]. For each
    connection, extracts traversed link indices, total physical lengths, cumulative span counts,
    and required inline amplifiers[cite: 10].

    Args:
    ---------
        G (nx.Graph): 
            Network topology graph with edge weights representing link physical length in km[cite: 10].
            Edges must contain an 'idx' attribute for link identification[cite: 10].
        core_nodes (List[Any]): 
            List of node identifiers considered as core/terminal nodes for demand generation[cite: 10].
        kSP (int, optional): 
            Maximum number of shortest path candidates to determine per node pair[cite: 10]. Default is 3[cite: 10].
        Lspan (float, optional): 
            Standard maximum fiber span length in km used for span partitioning[cite: 10]. Default is 80.0 km[cite: 10].

    Returns:
    ---------
        Tuple[nx.Graph, np.ndarray, np.ndarray]:
            - G (nx.Graph): Updated NetworkX graph containing 'weights_Nspan' and 'weights_Lspan'[cite: 10].
            - All_connections_Profile (np.ndarray): 2D object matrix of shape (num_connections, 7)[cite: 10],
              storing:
                * Column 0: Source core node[cite: 10]
                * Column 1: Destination core node[cite: 10]
                * Column 2: List of constituent edge 'idx' values per path[cite: 10]
                * Column 3: Total physical path lengths in km[cite: 10]
                * Column 4: Total span count per path[cite: 10]
                * Column 5: Sequences of nodes along each path[cite: 10]
                * Column 6: Total inline amplifiers per path[cite: 10]
            - degree_node_all_topo (np.ndarray): Array of nodal degrees for all nodes in graph G[cite: 10].

    Example:
    ---------
    >>> import networkx as nx
    >>> from CFM.utils.connection_profile import build_connection_profile
    >>> G = nx.Graph()
    >>> G.add_edge(1, 2, weight=120.0, idx=1)
    >>> G.add_edge(2, 3, weight=90.0, idx=2)
    >>> G.add_edge(1, 3, weight=250.0, idx=3)
    >>> core_nodes = [1, 2, 3]
    >>> G_mod, conn_profile, degrees = build_connection_profile(G, core_nodes, kSP=2, Lspan=80.0)
    """
    for u, v, data in G.edges(data=True):
        w = data['weight']
        data['weights_Nspan'] = np.ceil(w / Lspan)
        data['weights_Lspan'] = w / np.ceil(w / Lspan)

    nodes = list(G.nodes)
    nn = len(nodes)

    degree_node_all_topo = np.zeros(nn)
    Total_ROADM_Modul = 0

    for i, node in enumerate(nodes):
        degree_node_all_topo[i] = len(G[node])
        Total_ROADM_Modul += 2 * degree_node_all_topo[i]

    num_core = len(core_nodes)
    num_connections = int(0.5 * num_core * (num_core - 1))

    All_connections_Profile = np.empty((num_connections, 7), dtype=object)

    connection_counter = 0
    all_LP_length_km = []
    all_LP_Nspan = []

    for s_idx in range(num_core):
        source = core_nodes[s_idx]

        for d_idx in range(s_idx + 1, num_core):
            destination = core_nodes[d_idx]

            shortestPaths = np.empty((kSP,), dtype=object)
            totalCosts = np.zeros(kSP)

            x = nx.shortest_simple_paths(G, source, destination, weight='weight')
            number_of_paths = 0

            for i in range(kSP):
                try:
                    shortestPaths[i] = next(x)
                    totalCosts[i] = nx.path_weight(G, shortestPaths[i], weight='weight')
                    number_of_paths += 1
                except (StopIteration, nx.NetworkXNoPath):
                    break

            link_list_allpath = np.empty((number_of_paths,), dtype=object)
            totalCosts_allpath_Nspan = np.zeros(number_of_paths)
            Num_inLine_Amp = np.zeros(number_of_paths)

            for k in range(number_of_paths):
                path = shortestPaths[k]
                links = np.zeros(len(path) - 1)

                for i in range(len(path) - 1):
                    links[i] = G[path[i]][path[i + 1]]['idx']

                link_list_allpath[k] = links
                totalCosts_allpath_Nspan[k] = nx.path_weight(G, path, weight='weights_Nspan')
                Num_inLine_Amp[k] = totalCosts_allpath_Nspan[k] - len(path)

                all_LP_length_km.append(totalCosts[k])
                all_LP_Nspan.append(totalCosts_allpath_Nspan[k])

            All_connections_Profile[connection_counter, 0] = source
            All_connections_Profile[connection_counter, 1] = destination
            All_connections_Profile[connection_counter, 2] = link_list_allpath
            All_connections_Profile[connection_counter, 3] = totalCosts
            All_connections_Profile[connection_counter, 4] = totalCosts_allpath_Nspan
            All_connections_Profile[connection_counter, 5] = shortestPaths
            All_connections_Profile[connection_counter, 6] = Num_inLine_Amp + 1

            connection_counter += 1

    return G, All_connections_Profile, degree_node_all_topo