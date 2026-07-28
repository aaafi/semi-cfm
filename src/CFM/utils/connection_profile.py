import numpy as np
import networkx as nx

def build_connection_profile(G, core_nodes, kSP=3, Lspan=80):

  # ---- Add span information ----
  for u, v, data in G.edges(data=True):

      w = data['weight']

      data['weights_Nspan'] = np.ceil(w / Lspan)
      data['weights_Lspan'] = w / np.ceil(w / Lspan)

  # ---- Degree of nodes ----
  nodes = list(G.nodes)
  nn = len(nodes)

  degree_node_all_topo = np.zeros(nn)
  Total_ROADM_Modul = 0

  for i, node in enumerate(nodes):
      degree_node_all_topo[i] = len(G[node])
      Total_ROADM_Modul += 2 * degree_node_all_topo[i]

  # ---- Core node pairs ----
  num_core = len(core_nodes)
  num_connections = int(0.5 * num_core * (num_core - 1))

  All_connections_Profile = np.empty((num_connections, 7), dtype=object)

  connection_counter = 0
  all_LP_length_km = []
  all_LP_Nspan = []

  # ---- K shortest paths for each core pair ----
  for s_idx in range(num_core):

    source = core_nodes[s_idx]

    for d_idx in range(s_idx+1, num_core):

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
        except:
            break

      link_list_allpath = np.empty((number_of_paths,), dtype=object)
      totalCosts_allpath_Nspan = np.zeros(number_of_paths)
      Num_inLine_Amp = np.zeros(number_of_paths)

      for k in range(number_of_paths):

        path = shortestPaths[k]

        links = np.zeros(len(path)-1)

        for i in range(len(path)-1):
          links[i] = G[path[i]][path[i+1]]['idx']

        link_list_allpath[k] = links

        totalCosts_allpath_Nspan[k] = nx.path_weight(G, path, weight='weights_Nspan')

        Num_inLine_Amp[k] = totalCosts_allpath_Nspan[k] - len(path)

        all_LP_length_km.append(totalCosts[k])
        all_LP_Nspan.append(totalCosts_allpath_Nspan[k])

      # ---- Store connection profile ----
      All_connections_Profile[connection_counter,0] = source
      All_connections_Profile[connection_counter,1] = destination
      All_connections_Profile[connection_counter,2] = link_list_allpath
      All_connections_Profile[connection_counter,3] = totalCosts
      All_connections_Profile[connection_counter,4] = totalCosts_allpath_Nspan
      All_connections_Profile[connection_counter,5] = shortestPaths
      All_connections_Profile[connection_counter,6] = Num_inLine_Amp + 1

      connection_counter += 1

  return G, All_connections_Profile, degree_node_all_topo
