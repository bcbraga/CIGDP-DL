"""
Este código é uma implementação feita por Bruna Cristina Braga Charytitsch do método GRASP_SDNE*: Cb_sdne* + Local Search.
O método para construção de uma solução inicial G_SDNE* utilizando a técnica SDNE para geração de embeddings foi
proposto por Bruna em sua tese de doutorado. A implementação foi feita em linguagem Python para posterior comparação
deste método com o GRASP2 e GRASP3 da literatura, propostos por Napoletano et al. (2019)
<https://www.sciencedirect.com/science/article/abs/pii/S0377221718308701>.

Date: [26/06/2024]

This code is an implementation by Bruna Cristina Braga Charytitsch of the GRASP_SDNE*: Cb_sdne* + Local Search method.
The method for constructing an initial solution G_SDNE* using the SDNE technique for generating embeddings was
proposed by Bruna in her PhD thesis. The implementation was done in Python for subsequent comparison of this
method with the GRASP2 and GRASP3 from the literature, proposed by Napoletano et al. (2019)
<https://www.sciencedirect.com/science/article/abs/pii/S0377221718308701>.

Date: [06/26/2024]
"""


# BIBLIOTECAS/LIBRARIES

import time
from datetime import timedelta
import numpy as np
from networkx import DiGraph, parse_adjlist
from pandas import DataFrame
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import random
import collections as col

from cogdl.models.emb.sdne import SDNE

import torch
from cogdl.data import Graph


# BEGIN DATA READING:--------------------------------------------------------------##
def data_read():
    graph_id = []
    file_path = 'data/ids'  # /home/bruna/PycharmProjects/GraRep

    """
    The list graph_id contains the lines from the file "ids", without newline characters.
    Using "with open(...)" to ensure that the file is properly closed after reading.
    """
    with open(file_path, 'r') as file:
        for line in file:
            graph_id.append(line.strip())

    """
    Explanations:
    The list "all_graphs" is used to store data from all the files. Each element of the list contains three items:
    the "graph" array, the "nel" array, and the integer "nl". We use the "with" context manager to open and close the 
    files, avoiding leaving files open unnecessarily. During the file reading, we convert the strings to integers using
    "map", making the process more efficient. After the loop, we have the "all_graphs" list containing information from
    all the files in the form of NumPy arrays.
    """

    # List to store the data from the files.
    all_graphs = []

    for temp in graph_id:
        file_path = 'data/' + str.strip(temp) + '.txt'

        with open(file_path, 'r') as file:
            # Removing leading and trailing blank lines and converting to integers.
            lines = [list(map(int, line.split())) for line in file if line.strip()]

        # Checking if there is enough data in the file.
        if len(lines) >= 3:
            nl = lines[0][0]
            nel = lines[1]
            graph = lines[2:]

            df = DataFrame(graph).fillna(-1).astype(int)
            df_a = df.to_numpy(dtype=int)

            all_graphs.append((df_a, np.array(nel), nl))

    return all_graphs


# END DATA READING:--------------------------------------------------------------##

# FUNCTIONS------------------------------------------##

# Return quantity of incremental nodes from the first column of the vector
def cont(v1):
    return np.count_nonzero(v1 == 0)


# Return quantity of incremental nodes from the first column of the vector pos insertion incremental nodes
def cont2(v2):
    return np.count_nonzero(v2 == -1)


# Apply the mask and return the elements on the line - neighborhood
def masc(v3):
    return v3[v3 != -1]


# Return index after verifying if it is not an incremental node
def find1(el, x):
    return np.where(el == x)[0][0]


# Identify cross
def cross(u_, ul_, v_, vl_):
    return int((u_ < ul_ and v_ > vl_) or (u_ > ul_ and v_ < vl_))


# Given the graph in the initial format, generates an adjacency list and returns a DiGraph (NetworkX).
def make_graph(v4):
    a1 = v4.tolist()
    a2 = []
    for i in np.arange(len(a1)):
        t1 = str(a1[i]).strip('[]')
        t2 = t1.replace(',', ' ')
        a2.append(t2)
    g = parse_adjlist(a2, nodetype=int, create_using=DiGraph)

    # remove auxiliar -1 / remove auxiliary -1
    if -1 in g:
        g.remove_node(-1)

    return g


# Return the candidate list
def cand_list(cand, lay, elm):
    lista0 = [cont(cand[i:i + elm[i], 0]) if i == 0
              else cont(cand[np.sum(elm[0:i]):np.sum(elm[0:i]) + elm[i], 0]) for i in np.arange(lay)]
    max_node = max(lista0)
    lista = np.array(lista0)

    t = [j if j == 0 else np.sum(elm[0:j]) for j in np.arange(lay)]
    candidate = [[cand[t[j] + elm[j] - lista[j] + k0, 1] if (k0 < lista[j]) else -1 for k0 in range(max_node)] for j in
                 np.arange(lay)]

    return candidate


# Returns a permutation vector with original nodes in their initial positions and the remaining spaces with -1.
def perm2(v5):
    return list(map(lambda x, y: x * y if y != 0 else -1, v5[0:, 1], v5[0:, 0]))


def atualizap(no_, vet_, pos_no_):
    vet_[pos_no_ - 1], vet_[pos_no_] = no_, vet_[pos_no_ - 1]
    return vet_


# calculate the total of arcs crossings in the initial graph/
# Returns initial cross and an array with one digrafo (with atributtes: cam, id, pos_i, cx, cy) per layer.
def fc(graf, la, ne, inc, node_id, emb_r2):
    s_cross = 0

    # keep Digraphs generated
    gr = np.empty(la, dtype=object)

    for i in np.arange(la - 1):
        n = ne[i]
        aux = np.sum(ne[0:i]) if i > 0 else 0

        # compute the total of incremental nodes
        i_n = cont(graf[aux:aux + n, 0])
        gra = make_graph(graf[aux:aux + n - i_n:1, 1:])

        # Update arc list by removing successor nodes that have not been inserted yet.
        for uu, vv in list(gra.edges()):

            # attributes original nodes (only in the source nodes of arcs in the current layer)
            gra.nodes[uu]['id'] = 1
            gra.nodes[uu]['cam'] = i
            gra.nodes[uu]['pos_i'] = find1(uu, graf[aux:aux + n:1, 1])

            # Storing coordinates of embeddings
            if i == 0:
                gra.nodes[uu]['cx'] = emb_r2[np.where(node_id == uu), 0]
                gra.nodes[uu]['cy'] = emb_r2[np.where(node_id == uu), 1]
            if 0 < i < la - 1:
                gra.nodes[uu]['cx'] = emb_r2[np.where(node_id == uu + aux), 0]
                gra.nodes[uu]['cy'] = emb_r2[np.where(node_id == uu + aux), 1]

        # auxiliary graph
        gra2 = gra.copy()

        # removing incremental neighbors
        for u, v in list(gra.edges()):
            if v in inc[i + 1]:
                gra.remove_edge(u, v)

        for i1 in graf[aux + n - i_n:aux + n:1, 1]:
            for i2 in masc(graf[aux + find1(i1, graf[aux:aux + n:1, 1]), 2:]):
                gra2.add_edge(i1, i2)

            # incremental nodes atributes (head)
            gra2.nodes[i1]['id'] = 0
            gra2.nodes[i1]['cam'] = i
            gra2.nodes[i1]['pos_i'] = find1(i1, graf[aux:aux + n:1, 1])

            # Storing coordinates of embeddings
            if i == 0:
                gra2.nodes[i1]['cx'] = emb_r2[np.where(node_id == i1), 0]
                gra2.nodes[i1]['cy'] = emb_r2[np.where(node_id == i1), 1]
            if 0 < i < la - 1:
                gra2.nodes[i1]['cx'] = emb_r2[np.where(node_id == i1 + aux), 0]
                gra2.nodes[i1]['cy'] = emb_r2[np.where(node_id == i1 + aux), 1]

        # keep the head and tail nodes of each arc
        arst = np.array(gra.edges)

        # Keep the initial position of the head and tail nodes of each arc.
        a_pos = np.zeros((arst.shape[0], 2), dtype=int)

        for c1 in np.arange(arst.shape[0]):
            a_pos[c1, 0] = find1(arst[c1, 0], graf[aux:aux + n:1, 1:2:1])
            a_pos[c1, 1] = find1(arst[c1, 1], graf[aux + n:aux + n + ne[i + 1]:1, 1:2:1])

        # Computing crossings
        for k1 in np.arange(a_pos.shape[0] - 1):
            for j in np.arange(k1 + 1, a_pos.shape[0], 1):
                s_cross += cross(a_pos[k1][0], a_pos[j][0], a_pos[k1][1], a_pos[j][1])

        # Prepare array with la-1 grafos (1 por layer)
        gr[i] = gra2

    n0 = ne[la - 1]
    aux0 = sum(ne[0:la - 1])
    gra3 = DiGraph()

    for i3 in graf[aux0:aux0 + n0:1, 1]:

        a = find1(i3, graf[aux0:aux0 + n0:1, 1])
        # original nodes
        if graf[aux0 + a, 0] == 1:
            gra3.add_node(i3, id=1, cam=la - 1, pos_i=a, cx=emb_r2[np.where(node_id == i3 + aux0), 0],
                          cy=emb_r2[np.where(node_id == i3 + aux0), 1])
        # incremental nodes
        if graf[aux0 + a, 0] == 0:
            gra3.add_node(i3, id=0, cam=la - 1, pos_i=a, cx=emb_r2[np.where(node_id == i3 + aux0), 0],
                          cy=emb_r2[np.where(node_id == i3 + aux0), 1])

    # graph form the last layer
    gr[la - 1] = gra3

    return s_cross, gr


# Contribution - Mapping of nodes from the initial list
"""
The input instances of the graph are organized as follows: each node from a specific layer takes integer values 
from 0 to the total number of nodes in that layer - n. However, different layers will have nodes with the same IDs.
The "map" function transforms the IDs of each node considering its position in the layer it belongs to. This way, 
if Layer 1 has 10 elements, its nodes will have IDs from 0 to 9, and if Layer 2 has 20 nodes, their IDs will range
from 10 to 29. This approach allows us to work with the graph as a whole.
 """


def map_g(g0, nl0, nel0):
    g_new = np.copy(g0)

    # sweep layer
    for i0 in np.arange(nl0):
        n = nel0[i0]
        aux = np.sum(nel0[0:i0]) if i0 > 0 else 0

        # First layer
        if i0 == 0:
            non_empty_indices = (g0[aux:aux + n, 2:] != -1)
            g_new[aux:aux + n, 2:][non_empty_indices] = g0[aux:aux + n, 2:][non_empty_indices] + n

        # Middle layers
        if (i0 > 0) and (i0 < nl0 - 1):
            offset = np.sum(nel0[0:i0])
            g_new[aux:aux + n, 1] = g0[aux:aux + n, 1] + offset
            non_empty_indices = (g0[aux:aux + n, 2:] != -1)
            g_new[aux:aux + n, 2:][non_empty_indices] = g0[aux:aux + n, 2:][non_empty_indices] + sum(nel0[0:i0 + 1])

        # Last layer
        if i0 == nl0 - 1:
            offset = np.sum(nel0[0:i0])
            g_new[aux:aux + n, 1] = g0[aux:aux + n, 1] + offset

    # I generate a graph with updated IDs.
    """
    The Make_Graph function transforms the array containing graph information into an adjacency list, and subsequently
     this list is converted into a graph. Note that nodes in the array with the value -1 (empty) are removed from the 
     final graph.
    """
    # I pass the graph while ignoring the first row, which indicates whether a node is original or incremental.
    g1 = make_graph(g_new[:, 1:])
    # print(g_new)

    return g1


# Returns embeddings in R^2 (two-dimensional space).
def plan_emb(res_emb, dado):
    # Retrieve node embeddings and corresponding subjects
    node_embeddings = (
        res_emb
    )

    transform = PCA
    trans = transform(n_components=2)
    node_embeddings_2d = trans.fit_transform(node_embeddings[dado.nodes])

    return node_embeddings_2d


# embeddings visualization
def visual_emb(node_embeddings_2d, dad):
    transform = PCA
    # draw the embedding points, coloring them by the target label (paper subject)
    alpha = 0.7
    # label_map = {l: i for i, l in enumerate(unique(dad.nodes))}

    node_colours = [0 if (target < 10) else 1 for target in dad.nodes]

    plt.figure(figsize=(7, 7))
    plt.axes().set(aspect="equal")
    plt.scatter(
        node_embeddings_2d[:, 0],
        node_embeddings_2d[:, 1],
        c=node_colours,
        cmap="jet",
        alpha=alpha,
    )
    plt.title("{} visualization of node embeddings".format(transform.__name__))
    # Plot graph in the plane
    plt.show()

    return


# Data preprocessing and generation of embeddings.
def pre_emb_data(graph0, nl0, nel0):

    # organize indexes
    dado0 = map_g(graph0, nl0, nel0)

    # putting data in cogdl format
    edges = torch.tensor(np.array(dado0.edges())).t()
    g_cogdl = Graph(edge_index=edges)

    # SDNE(hidden_size1, hidden_size2, droput, alpha, beta, nu1, nu2, epochs, lr, cpu)
    model = SDNE(1000, 128, 0.5, 0.01, 5, 0.0001, 0.001, 500, 0.01, "store_true")  # True

    res_emb0 = model.forward(g_cogdl)

    # The order of embedding vectors follows the order of g_cogdl.nodes
    # Returns node indices in the order of embeddings.
    node_ids0 = np.array(g_cogdl.nodes())

    # Embedding coordinates in the plane.
    emb_2d0 = plan_emb(res_emb0, dado0)

    # visualize the embedding in the plane
    # visual_emb(emb_2d0, dado0)

    return node_ids0, emb_2d0


# calculates the Euclidean distance
def dist_euc_2d(x1, y1, x2, y2):
    a0 = x2 - x1
    b0 = y2 - y1
    c0 = np.sqrt(np.power(a0, 2) + np.power(b0, 2))
    return c0


# Initialize variables.
def begin_f(graph_, nel_, nl_, node_ids_, emb_2d_):
    # construction of candidate list / Array with incremental nodes - rows represent layers
    cl_ = np.array(cand_list(graph_[0:, 0:2:1], nl_, nel_))

    # Returns the initial crossing and nl-1 digraphs (NetworkX)
    ini_cross_, grafos_ = fc(graph_, nl_, nel_, cl_, node_ids_, emb_2d_)

    # permutation vector
    per_ = np.array(perm2(graph_[0:, 0:2:1]))

    return cl_, ini_cross_, grafos_, per_


# Function assigns Euclidean distance to graph edges.
def ares_atr(g2, nl_2):
    # Sweeps through layers.
    for i in np.arange(nl_2 - 1):
        for uu, vv in list(g2[i].edges()):
            x1_ = g2[i].nodes[uu]['cx']
            y1_ = g2[i].nodes[uu]['cy']
            x2_ = g2[i + 1].nodes[vv]['cx']
            y2_ = g2[i + 1].nodes[vv]['cy']
            # computing the Euclidian distance
            g2[i].edges[uu, vv]['dis'] = dist_euc_2d(x1_, y1_, x2_, y2_)

    return g2


# Identifies and returns the minimum distances, along with the position (max and min) of the corresponding neighbor
def func_max_min(gra9, eds9, i9, vet9, id_l):
    # Initialize variables
    min_dis_ = float('inf')
    max_dis_ = -1
    # Current position of node in P (neighbors of no_in) belonging to the edge with min/max distance
    min_pos_viz_ = None
    max_pos_viz_ = None

    # Access and print attributes of specific edges
    for edge in eds9:
        node1, node2 = edge
        attrib = gra9[i9].edges[edge]['dis']

        if attrib < min_dis_:
            min_dis_ = attrib
            # Previous layer to the next layer
            if id_l == 0:
                min_pos_viz_ = find1(node2, vet9)
            # Subsequent layer to the previous layer
            if id_l == 1:
                min_pos_viz_ = find1(node1, vet9)

        if attrib > max_dis_:
            max_dis_ = attrib
            # Previous layer to the next layer
            if id_l == 0:
                max_pos_viz_ = find1(node2, vet9)
            # Subsequent layer to the previous layer
            if id_l == 1:
                max_pos_viz_ = find1(node1, vet9)

    return min_dis_, min_pos_viz_, max_pos_viz_


# Returns one of the positions - near the maximum, minimum, or in between.
def func_ale(min_pos_viz_1, max_pos_viz_1):

    res1 = None
    # Inserting a random component
    opc = random.randint(0, 2)
    # Position near the neighbor with the smallest distance
    if opc == 0:
        res1 = min_pos_viz_1
    # position near the neighbor with the largest distance
    if opc == 1:
        res1 = max_pos_viz_1
    # position between the average positions of neighbors with the greatest and smallest distances
    if opc == 2:
        res1 = (min_pos_viz_1 + max_pos_viz_1) // 2

    return res1


# Returns the highest value of the calculated 'dis' attribute
def max_dis(grafos_r, nl_r):

    m_d = 0.0
    for i5 in np.arange(nl_r):
        for u, v, attrs in grafos_r[i5].edges(data=True):
            if 'dis' in attrs:
                m_val = grafos_r[i5].edges[u, v]['dis']
                if m_val > m_d:
                    m_d = m_val

    return m_d


def pre_const_phase_cb(grafos1, nl1, nel1, per1, cl1, alp1):
    # Generates matrices (layers x elements) - rows = layers - columns = incremental nodes (blank spaces = -1)
    # d_min = minimum found distance / d_min_pos = position for minimum d

    # (1) Computing d(v)

    # Vectorized Version 1
    d_min = (-1) * np.ones((nl1, cl1.shape[1]), dtype=float)
    d_min_pos = (-1) * np.ones((nl1, cl1.shape[1]), dtype=int)

    # sweep layers
    for i in np.arange(nl1):
        n = nel1[i]
        aux = sum(nel1[0:i]) if i > 0 else 0

        # sweep incremental nodes
        for j in np.arange(cl1.shape[1]):
            no_in = cl1[i, j]
            if no_in != -1:

                # node belongs to the first layer.
                if i == 0:
                    # check if node has neighbors
                    dep = list(grafos1[i].neighbors(no_in))
                    # Check if the neighbors have been added to the graph
                    tem_e = any(ele in dep for ele in per1[aux + n:aux + n + nel1[i + 1]])

                    # No neighbors present or not inserted
                    # Not a good candidate for the rcl /Insert in the last available position or at a random position
                    if (not dep) or (dep and not tem_e):

                        # last available position
                        # d_min_pos[i][j] = list(per1[aux:aux + n]).index(-1)

                        # random position
                        d_min_pos[i][j] = random.randint(0, n - 1)
                        # The largest calculated distance is assigned
                        d_min[i][j] = max_dis(grafos1, nl1)

                    else:
                        # stores the already inserted neighbor in p
                        temp_p = per1[aux + n:aux + n + nel1[i + 1]]
                        neigh = np.array([dep[z] for z in np.arange(len(dep)) if (dep[z] in temp_p)])

                        # List of edges leaving the node
                        # eds1 = grafos1[i].out_edges(no_in)

                        # List of edges leaving the source node and going to the specific destination nodes
                        eds = [(no_in, tail_no) for tail_no in neigh if grafos1[i].has_edge(no_in, tail_no)]

                        d_min[i][j], min_pos_viz, max_pos_viz = func_max_min(grafos1, eds, i, temp_p, 0)
                        # returns one of the positions - near the maximum, minimum, or in between.
                        d_min_pos[i][j] = func_ale(min_pos_viz, max_pos_viz)

                # node belongs to the last layer.
                if i == nl1 - 1:

                    # check if node has neighbors
                    ant = list(grafos1[i - 1].predecessors(no_in))
                    # Check if the neighbors have been added to the graph
                    tem_e = any(ele in ant for ele in per1[aux - nel1[i - 1]:aux])

                    # No neighbors present or not inserted
                    # Not a good candidate for the rcl /Insert in the last available position or at a random position
                    if (not ant) or (ant and not tem_e):

                        # last available position
                        # d_min_pos[i][j] = list(per1[aux:aux + n]).index(-1)

                        # random position
                        d_min_pos[i][j] = random.randint(0, n - 1)
                        d_min[i][j] = max_dis(grafos1, nl1)

                    else:
                        # stores the already inserted neighbor in p
                        temp_p = per1[aux - nel1[i - 1]:aux]
                        neigh = np.array([ant[z] for z in np.arange(len(ant)) if (ant[z] in temp_p)])

                        # List of edges incoming the node
                        # eds1 = grafos1[i].in_edges(no_in)

                        # List of edges leaving the destination node and going to the specific sources nodes
                        eds = [(head_no, no_in) for head_no in neigh if grafos1[i - 1].has_edge(head_no, no_in)]

                        d_min[i][j], min_pos_viz, max_pos_viz = func_max_min(grafos1, eds, i - 1, temp_p, 1)
                        # returns one of the positions - near the maximum, minimum, or in between.
                        d_min_pos[i][j] = func_ale(min_pos_viz, max_pos_viz)

                # node belongs to a middle layer
                if 0 < i < nl1 - 1:

                    # check if node has neighbors

                    dep = list(grafos1[i].neighbors(no_in))
                    # Check if the neighbors have been added to the graph
                    tem_e_d = any(ele in dep for ele in per1[aux + n:aux + n + nel1[i + 1]])

                    ant = list(grafos1[i - 1].predecessors(no_in))
                    # Check if the neighbors have been added to the graph
                    tem_e_a = any(ele in ant for ele in per1[aux - nel1[i - 1]:aux])

                    # They do not have previous or subsequent neighbors
                    if dep == [] and ant == []:

                        # last available position
                        # d_min_pos[i][j] = list(per1[aux:aux + n]).index(-1)

                        # random position
                        d_min_pos[i][j] = random.randint(0, n - 1)
                        d_min[i][j] = max_dis(grafos1, nl1)

                    elif (tem_e_d is False) and (tem_e_a is False):
                        # last available position
                        # d_min_pos[i][j] = list(per1[aux:aux + n]).index(-1)

                        # random position
                        d_min_pos[i][j] = random.randint(0, n - 1)
                        d_min[i][j] = max_dis(grafos1, nl1)  # 0

                    else:

                        # It only has neighbors in the next layer
                        if (tem_e_a is False) and (tem_e_d is True):
                            # stores the already inserted neighbor in p
                            temp_p = per1[aux + n:aux + n + nel1[i + 1]]
                            neigh = np.array([dep[z] for z in np.arange(len(dep)) if (dep[z] in temp_p)])

                            # List of edges leaving the node
                            # eds1 = grafos1[i].out_edges(no_in)

                            # List of edges leaving the source node and going to the specific destination nodes
                            eds = [(no_in, tail_no) for tail_no in neigh if grafos1[i].has_edge(no_in, tail_no)]

                            d_min[i][j], min_pos_viz, max_pos_viz = func_max_min(grafos1, eds, i, temp_p, 0)
                            # returns one of the positions - near the maximum, minimum, or in between.
                            d_min_pos[i][j] = func_ale(min_pos_viz, max_pos_viz)

                        # It only has neighbors in the previous layer
                        if (tem_e_d is False) and (tem_e_a is True):
                            # stores the already inserted neighbor in p
                            temp_p = per1[aux - nel1[i - 1]:aux]
                            neigh = np.array([ant[z] for z in np.arange(len(ant)) if (ant[z] in temp_p)])

                            # List of edges incoming the node
                            # eds1 = grafos1[i].in_edges(no_in)

                            # List of edges leaving the destination node and going to the specific sources nodes
                            eds = [(head_no, no_in) for head_no in neigh if grafos1[i - 1].has_edge(head_no, no_in)]

                            d_min[i][j], min_pos_viz, max_pos_viz = func_max_min(grafos1, eds, i - 1, temp_p, 1)
                            # returns one of the positions - near the maximum, minimum, or in between.
                            d_min_pos[i][j] = func_ale(min_pos_viz, max_pos_viz)

                        # It has neighbors in both, previous and next layer
                        if (tem_e_a is True) and (tem_e_d is True):
                            # next
                            # stores the already inserted neighbor in p
                            temp_pa = per1[aux + n:aux + n + nel1[i + 1]]
                            neigh_a = np.array([dep[z] for z in np.arange(len(dep)) if (dep[z] in temp_pa)])

                            # List of edges leaving the node
                            # eds1 = grafos1[i].out_edges(no_in)

                            # List of edges leaving the source node and going to the specific destination nodes
                            eds_a = [(no_in, tail_no) for tail_no in neigh_a if grafos1[i].has_edge(no_in, tail_no)]

                            min_a, min_pos_viz_a, max_pos_viz_a = func_max_min(grafos1, eds_a, i, temp_pa, 0)
                            # returns one of the positions - near the maximum, minimum, or in between.
                            d_min_pos_a = func_ale(min_pos_viz_a, max_pos_viz_a)

                            # previous
                            # stores the already inserted neighbor in p
                            temp_pb = per1[aux - nel1[i - 1]:aux]
                            neigh_b = np.array([ant[z] for z in np.arange(len(ant)) if (ant[z] in temp_pb)])

                            # List of edges incoming the node
                            # eds1 = grafos1[i].in_edges(no_in)

                            # List of edges leaving the destination node and going to the specific sources nodes
                            eds_b = [(head_no, no_in) for head_no in neigh_b if grafos1[i - 1].has_edge(head_no, no_in)]

                            min_b, min_pos_viz_b, max_pos_viz_b = func_max_min(grafos1, eds_b, i - 1, temp_pb, 1)
                            # returns one of the positions - near the maximum, minimum, or in between.
                            d_min_pos_b = func_ale(min_pos_viz_b, max_pos_viz_b)

                            # ---

                            d_min[i][j] = min(min_a, min_b)
                            d_min_pos[i][j] = func_ale(d_min_pos_a, d_min_pos_b)

    # computing global max and min
    min_g = min(d_min[d_min != -1])
    max_g = max(d_min[d_min != -1])

    # computing tau to all incremental node in candidate list and construct RCL
    min_threshold = min_g + alp1 * (max_g - min_g)
    rcl_ = np.where((d_min != -1) & (d_min <= min_threshold), cl1, -1)

    return d_min, d_min_pos, rcl_


# Check if the position can be filled with 'node'
def check_position(grafos4, k4, layer4, no4, cont4, aux4, i4):
    node_info = grafos4[layer4].nodes[aux4[i4 - 1]]
    id_value = node_info['id']

    # incremental node
    if id_value == 0:
        aux4 = atualizap(no4, aux4, i4)
        cont4 += 1
    # original node
    elif id_value == 1:
        pos_i = node_info['pos_i']
        lower_bound = max(0, pos_i - k4)
        upper_bound = min(pos_i + k4, aux4.shape[0] - 1)

        if lower_bound <= i4 <= upper_bound:
            aux4 = atualizap(no4, aux4, i4)
            cont4 += 1

    return cont4, aux4


# return the position to insert a node give the node and a specific position to insert it
def g_vq(grafos3, nel3, per3, k3, lay3, no3, q3):
    pos_final = None
    n = nel3[lay3]
    aux = np.sum(nel3[0:lay3]) if lay3 > 0 else 0

    # Check if there is space in the layer.
    if -1 in per3[aux:aux + n]:

        aux1 = np.copy(per3[aux:aux + n])
        ind = aux1.shape[0] - col.Counter(per3[aux:aux + n])[-1]
        aux1[ind] = no3

        if q3 >= ind:
            pos_final = ind
        else:
            con = 0
            for i3 in np.arange(ind, q3, -1):
                cont3 = 0
                (cont3, aux1) = check_position(grafos3, k3, lay3, no3, cont3, aux1, i3)
                # If a constraint is violated, the loop is interrupted
                if cont3 == 0:
                    break
                else:
                    con += 1

            # I can insert at the desired position
            if con == ind - q3:
                pos_final = q3

            # I cannot insert at the desired position
            else:
                pos_final = ind - con

    return pos_final


# THE FOLLOW CALL - Return RCL, min pos, min cross and p updated
def pre_const_phase_cb_2(grafos5, nl5, nel5, per5, cl5, alp5, layers5, d_min5, d_min_pos5):
    # sweep layers
    for i in layers5:
        n = nel5[i]
        aux = np.sum(nel5[0:i]) if i > 0 else 0

        # sweep incremental nodes
        for j in np.arange(cl5.shape[1]):
            no_in = cl5[i, j]
            if no_in != -1:

                # node belongs to the first layer.
                if i == 0:
                    # check if node has neighbors
                    dep = list(grafos5[i].neighbors(no_in))
                    # Check if the neighbors have been added to the graph
                    tem_e = any(ele in dep for ele in per5[aux + n:aux + n + nel5[i + 1]])

                    # No neighbors present or not inserted
                    # Not a good candidate for the rcl /Insert in the last available position or at a random position
                    if (not dep) or (dep and not tem_e):

                        # last available position
                        # d_min_pos[i][j] = list(per1[aux:aux + n]).index(-1)

                        # random position
                        d_min_pos5[i][j] = random.randint(0, n - 1)
                        d_min5[i][j] = max_dis(grafos5, nl5)

                    else:
                        # stores the already inserted neighbor in p
                        temp_p = per5[aux + n:aux + n + nel5[i + 1]]
                        neigh = np.array([dep[z] for z in np.arange(len(dep)) if (dep[z] in temp_p)])

                        # List of edges leaving the node
                        # eds1 = grafos1[i].out_edges(no_in)

                        # List of edges leaving the source node and going to the specific destination nodes
                        eds = [(no_in, tail_no) for tail_no in neigh if grafos5[i].has_edge(no_in, tail_no)]

                        d_min5[i][j], min_pos_viz, max_pos_viz = func_max_min(grafos5, eds, i, temp_p, 0)
                        # returns one of the positions - near the maximum, minimum, or in between.
                        d_min_pos5[i][j] = func_ale(min_pos_viz, max_pos_viz)

                # node belongs to the last layer.
                if i == nl5 - 1:

                    # check if node has neighbors
                    ant = list(grafos5[i - 1].predecessors(no_in))
                    # Check if the neighbors have been added to the graph
                    tem_e = any(ele in ant for ele in per5[aux - nel5[i - 1]:aux])

                    # No neighbors present or not inserted
                    # Not a good candidate for the rcl /Insert in the last available position or at a random position
                    if (not ant) or (ant and not tem_e):

                        # last available position
                        # d_min_pos[i][j] = list(per1[aux:aux + n]).index(-1)

                        # random position
                        d_min_pos5[i][j] = random.randint(0, n - 1)
                        d_min5[i][j] = max_dis(grafos5, nl5)

                    else:
                        # stores the already inserted neighbor in p
                        temp_p = per5[aux - nel5[i - 1]:aux]
                        neigh = np.array([ant[z] for z in np.arange(len(ant)) if (ant[z] in temp_p)])

                        # List of edges incoming the node
                        # eds1 = grafos1[i].in_edges(no_in)

                        # List of edges leaving the destination node and going to the specific sources nodes
                        eds = [(head_no, no_in) for head_no in neigh if grafos5[i - 1].has_edge(head_no, no_in)]

                        d_min5[i][j], min_pos_viz, max_pos_viz = func_max_min(grafos5, eds, i - 1, temp_p, 1)
                        # returns one of the positions - near the maximum, minimum, or in between.
                        d_min_pos5[i][j] = func_ale(min_pos_viz, max_pos_viz)

                # node belongs to a middle layer
                if 0 < i < nl5 - 1:

                    # check if node has neighbors

                    dep = list(grafos5[i].neighbors(no_in))
                    # Check if the neighbors have been added to the graph
                    tem_e_d = any(ele in dep for ele in per5[aux + n:aux + n + nel5[i + 1]])

                    ant = list(grafos5[i - 1].predecessors(no_in))
                    # Check if the neighbors have been added to the graph
                    tem_e_a = any(ele in ant for ele in per5[aux - nel5[i - 1]:aux])

                    # They do not have previous or subsequent neighbors
                    if dep == [] and ant == []:

                        # last available position
                        # d_min_pos[i][j] = list(per1[aux:aux + n]).index(-1)

                        # random position
                        d_min_pos5[i][j] = np.randint(0, n - 1)
                        d_min5[i][j] = max_dis(grafos5, nl5)

                    elif (tem_e_d is False) and (tem_e_a is False):
                        # last available position
                        # d_min_pos[i][j] = list(per1[aux:aux + n]).index(-1)

                        # random position
                        d_min_pos5[i][j] = random.randint(0, n - 1)
                        d_min5[i][j] = max_dis(grafos5, nl5)

                    else:

                        # It only has neighbors in the next layer
                        if (tem_e_a is False) and (tem_e_d is True):
                            # stores the already inserted neighbor in p
                            temp_p = per5[aux + n:aux + n + nel5[i + 1]]
                            neigh = np.array([dep[z] for z in np.arange(len(dep)) if (dep[z] in temp_p)])

                            # List of edges leaving the node
                            # eds1 = grafos1[i].out_edges(no_in)

                            # List of edges leaving the source node and going to the specific destination nodes
                            eds = [(no_in, tail_no) for tail_no in neigh if grafos5[i].has_edge(no_in, tail_no)]

                            d_min5[i][j], min_pos_viz, max_pos_viz = func_max_min(grafos5, eds, i, temp_p, 0)
                            # returns one of the positions - near the maximum, minimum, or in between.
                            d_min_pos5[i][j] = func_ale(min_pos_viz, max_pos_viz)

                        # It only has neighbors in the previous layer
                        if (tem_e_d is False) and (tem_e_a is True):
                            # stores the already inserted neighbor in p
                            temp_p = per5[aux - nel5[i - 1]:aux]
                            neigh = np.array([ant[z] for z in np.arange(len(ant)) if (ant[z] in temp_p)])

                            # List of edges incoming the node
                            # eds1 = grafos1[i].in_edges(no_in)

                            # List of edges leaving the destination node and going to the specific sources nodes
                            eds = [(head_no, no_in) for head_no in neigh if grafos5[i - 1].has_edge(head_no, no_in)]

                            d_min5[i][j], min_pos_viz, max_pos_viz = func_max_min(grafos5, eds, i - 1, temp_p, 1)
                            # returns one of the positions - near the maximum, minimum, or in between.
                            d_min_pos5[i][j] = func_ale(min_pos_viz, max_pos_viz)

                        # It has neighbors in both, previous and next layer
                        if (tem_e_a is True) and (tem_e_d is True):
                            # next
                            # stores the already inserted neighbor in p
                            temp_pa = per5[aux + n:aux + n + nel5[i + 1]]
                            neigh_a = np.array([dep[z] for z in np.arange(len(dep)) if (dep[z] in temp_pa)])

                            # List of edges leaving the node
                            # eds1 = grafos1[i].out_edges(no_in)

                            # List of edges leaving the source node and going to the specific destination nodes
                            eds_a = [(no_in, tail_no) for tail_no in neigh_a if grafos5[i].has_edge(no_in, tail_no)]

                            min_a, min_pos_viz_a, max_pos_viz_a = func_max_min(grafos5, eds_a, i, temp_pa, 0)
                            # returns one of the positions - near the maximum, minimum, or in between.
                            d_min_pos_a = func_ale(min_pos_viz_a, max_pos_viz_a)

                            # previous
                            # stores the already inserted neighbor in p
                            temp_pb = per5[aux - nel5[i - 1]:aux]
                            neigh_b = np.array([ant[z] for z in np.arange(len(ant)) if (ant[z] in temp_pb)])

                            # List of edges incoming the node
                            # eds1 = grafos1[i].in_edges(no_in)

                            # List of edges leaving the destination node and going to the specific sources nodes
                            eds_b = [(head_no, no_in) for head_no in neigh_b if grafos5[i - 1].has_edge(head_no, no_in)]

                            min_b, min_pos_viz_b, max_pos_viz_b = func_max_min(grafos5, eds_b, i - 1, temp_pb, 1)
                            # returns one of the positions - near the maximum, minimum, or in between.
                            d_min_pos_b = func_ale(min_pos_viz_b, max_pos_viz_b)

                            # ---

                            d_min5[i][j] = min(min_a, min_b)
                            d_min_pos5[i][j] = func_ale(d_min_pos_a, d_min_pos_b)

    # computing global max and min / Exclude the value -1
    masked_array = np.ma.masked_equal(d_min5, -1)
    # Calculate the minimum and maximum value among the non-masked values
    min_g5 = np.min(masked_array)
    max_g5 = np.max(masked_array)

    # (2) computing tau to all incremental node in candidate list and construct RCL
    rcl_mask = np.logical_and(d_min5 != -1, d_min5 <= min_g5 + alp5 * (max_g5 - min_g5))
    rcl_values = np.where(rcl_mask, cl5, -1)
    rcl5 = rcl_values

    return d_min5, d_min_pos5, rcl5, per5


# Construction of an initial solution
def const_phase_cb(grafos2, nl2, nel2, per2, cl2, k2, alp2, d_min2, d_min_pos2, rcl2):
    while -1 in per2:

        # select node v* randomly in RCL
        """
        This code finds the indices where rcl2 is not equal to -1 using np.where(). Then, it checks if there are any 
        non-empty indices. If there are, it selects a random index from those non-empty indices and uses it to access 
        the corresponding position in the rcl2 array.
        """

        # initialization
        x = None
        y = None
        # Find non-empty indices
        non_empty_indices = np.where(rcl2 != -1)

        if non_empty_indices[0].size > 0:
            random_index = np.random.randint(non_empty_indices[0].size)
            x = non_empty_indices[0][random_index]
            y = non_empty_indices[1][random_index]

        # I identify the closest available position to the requested one
        best_pos_min = g_vq(grafos2, nel2, per2, k2, x, cl2[x][y], d_min_pos2[x][y])
        # I update the position where the node will be inserted
        d_min_pos2[x][y] = best_pos_min

        # identify the original position of node cd[x][y] in Gr/p
        n = nel2[x]
        aux = np.sum(nel2[0:x]) if x > 0 else 0

        # -----
        # Use arrays numpy
        # Find the index of the first -1 in the list.
        ind = np.where(per2[aux:aux + n] == -1)[0][0]
        # Create a copy of per2 to prevent unwanted modifications to the original array
        temp_p = np.copy(per2)
        # Remove the first occurrence of -1 in the layer of the list
        temp_p = np.delete(temp_p, aux + ind)
        # Calculate the position where the new element will be inserted.
        insert_position = d_min_pos2[x][y] + aux
        # Insert the new element at the calculated position and update p
        up_p = np.insert(temp_p, insert_position, cl2[x][y])
        # -----

        # update cd
        cl2[x][y] = -1
        # updates:
        d_min2[x][y] = -1
        d_min_pos2[x][y] = -1

        # recompute g and tau, rebuild RCL
        if x == 0:
            (d_min2, d_min_pos2, rcl2, per2) = pre_const_phase_cb_2(grafos2, nl2, nel2, up_p, cl2, alp2,
                                                                    [x, x + 1], d_min2, d_min_pos2)

        # Last Layer (update previous and current layers)
        if x == nl2 - 1:
            (d_min2, d_min_pos2, rcl2, per2) = pre_const_phase_cb_2(grafos2, nl2, nel2, up_p, cl2, alp2,
                                                                    [x - 1, x], d_min2, d_min_pos2)

        # middle layer (update layer previous and current layer)
        if (x != 0) and (x != nl2 - 1):
            (d_min2, d_min_pos2, rcl2, per2) = pre_const_phase_cb_2(grafos2, nl2, nel2, up_p, cl2, alp2,
                                                                    [x - 1, x, x + 1], d_min2, d_min_pos2)

    return per2


def comp_cro(grafos7, p1_7, p2_7, u7, v7, cro7):
    for u, v in grafos7.edges():
        u7l = find1(u, p1_7)
        v7l = find1(v, p2_7)
        if (u7 < u7l and v7 > v7l) or (u7 > u7l and v7 < v7l):
            cro7 += 1
    return cro7


# Returns the total number of arc crossings final
# retorna o total de cruzamentos final
def cro_total_final_(grafos7, nl7, nel7, p7):

    cont_sa = 0

    # Percorre todos os layers / Sweep layers
    for i in np.arange(nl7 - 1):
        n = nel7[i]
        aux = np.sum(nel7[0:i]) if i > 0 else 0

        # Guarda arestas (origens, destinos) do grafo / Storing  graph edges (sources, destinations).
        # garante que no possui vizinhos / Ensures that the node has neighbors.
        origen = np.array([nodo for nodo in grafos7[i].nodes() if grafos7[i].neighbors(nodo) != []])
        destin = np.array([nodo for nodo in grafos7[i + 1].nodes() if grafos7[i].predecessors(nodo) != []])

        # arestas cujos no origens estao no conjunto origen / arestas cujos nos destinos estao no conjunto destin
        # Edges whose head nodes are in the head set / Edges whose tail nodes are in the tail set.
        dat_o = list(grafos7[i].edges(origen))
        dat_d = list(grafos7[i].in_edges(destin))

        """
        Para unir duas listas em Python sem repetir os elementos, você pode converter as listas para conjuntos (sets) e,
        em seguida, utilizar a operação de união (union) dos conjuntos. O conjunto é uma estrutura de dados que não 
        permite elementos duplicados, portanto, quando você unir os dois conjuntos, os elementos repetidos serão 
        eliminados automaticamente. Em seguida, você pode converter o resultado de volta para uma lista, se necessário.       

        To merge two lists in Python without repeating elements, you can convert the lists to sets and then use the set 
        union operation. A set is a data structure that doesn't allow duplicate elements, so when you merge the two 
        sets, duplicate elements will be automatically eliminated. Then, you can convert the result back to a list if 
        needed.
        """

        # Convertendo as listas em conjuntos e realizando a união
        # Converting the lists into sets and performing the union
        conjunto_resultante = set(dat_o).union(dat_d)

        # Convertendo o conjunto resultante de volta para uma lista unida - lis_uni
        # Converting the resulting set back into a merged list - lis_uni
        lis_uni = np.array(list(conjunto_resultante))

        # print(lis_uni)
        # print(' ')
        # print('p7[aux:aux + n])=', p7[aux:aux + n])
        # print(' ')
        # print('p7[aux + n:aux + n + nel7[i + 1]])=',p7[aux + n:aux + n + nel7[i + 1]])
        # print(' ')

        # guarda a posicao dos vertices nos respectivos vetores
        # Stores the positions of the vertices in their respective arrays.
        u_values = [find1(node[0], p7[aux:aux + n]) for node in lis_uni]
        # print('u_values =', u_values)
        v_values = [find1(node[1], p7[aux + n:aux + n + nel7[i + 1]]) for node in lis_uni]
        # print('v_values =', v_values)

        # compara arestas e contabiliaza os cruzamentos / Compares edges and tallies the intersections.
        for ii in range(len(lis_uni) - 1):
            u1 = u_values[ii]
            v1 = v_values[ii]
            for jj in range(ii + 1, len(lis_uni)):
                u2 = u_values[jj]
                v2 = v_values[jj]
                if (u1 < u2 and v1 > v2) or (u1 > u2 and v1 < v2):
                    cont_sa += 1

    return cont_sa


# Returns number of arc crossings after the inserting of incremental nodes (It is Ok)
# retorna cruzamentos inseridos pela insercao de nos incrementais - (CORRETO)
def cro_parc_final(grafos7, nl7, nel7, p7):

    cont_sa = 0

    # Percorre todos os layers / Sweep layers
    for i in np.arange(nl7 - 1):
        n = nel7[i]
        aux = np.sum(nel7[0:i]) if i > 0 else 0

        # Guarda arestas (origens, destinos) do grafo / Storing  graph edges (sources, destinations).
        # garante que no possui vizinhos / Ensures that the node has neighbors.

        # Incremental nodes
        origen = np.array([nodo for nodo, atri in grafos7[i].nodes(data='id') if atri == 0
                           and grafos7[i].neighbors(nodo) != []])
        destin = np.array([nodo for nodo, atri in grafos7[i + 1].nodes(data='id') if atri == 0
                           and grafos7[i].predecessors(nodo) != []])

        # Ortiginal nodes
        origen_ = np.array([nodo for nodo, atri in grafos7[i].nodes(data='id') if atri == 1
                           and grafos7[i].neighbors(nodo) != []])
        destin_ = np.array([nodo for nodo, atri in grafos7[i + 1].nodes(data='id') if atri == 1
                           and grafos7[i].predecessors(nodo) != []])

        # arestas cujos no origens estao no conjunto origen / arestas cujos nos destinos estao no conjunto destin
        # Edges whose head nodes are in the head set / Edges whose tail nodes are in the tail set.
        # Incremental nodes
        dat_o = list(grafos7[i].edges(origen))
        dat_d = list(grafos7[i].in_edges(destin))
        # Original nodes
        dat_o_ = list(grafos7[i].edges(origen_))
        dat_d_ = list(grafos7[i].in_edges(destin_))

        """
        Para unir duas listas em Python sem repetir os elementos, você pode converter as listas para conjuntos (sets) e,
        em seguida, utilizar a operação de união (union) dos conjuntos. O conjunto é uma estrutura de dados que não 
        permite elementos duplicados, portanto, quando você unir os dois conjuntos, os elementos repetidos serão 
        eliminados automaticamente. Em seguida, você pode converter o resultado de volta para uma lista, se necessário.       

        To merge two lists in Python without repeating elements, you can convert the lists to sets and then use the set 
        union operation. A set is a data structure that doesn't allow duplicate elements, so when you merge the two 
        sets, duplicate elements will be automatically eliminated. Then, you can convert the result back to a list if 
        needed.
        """

        # Convertendo as listas em conjuntos e realizando a união
        # Converting the lists into sets and performing the union
        # Incremental nodes
        conjunto_resultante = set(dat_o).union(dat_d)
        # Original nodes
        conjunto_resultante_ = set(dat_o_).union(dat_d_)

        # elimina elementos em comum / remove elements in both sets
        # Only original edges (arcs)
        conjunto_resultante_ -= conjunto_resultante

        # Convertendo o conjunto resultante de volta para uma lista unida - lis_uni
        # Converting the resulting set back into a merged list - lis_uni

        # incrementais
        lis_uni = np.array(list(conjunto_resultante))

        # originais
        lis_uni_ = np.array(list(conjunto_resultante_))

        # lista concatenada / concatened list where the first part contains the incremental nodes and after the original
        lis_final = np.concatenate((lis_uni, lis_uni_))

        # print('lista_final=', lis_final)
        # print(lis_uni)
        # print(' ')
        # print(lis_uni_)
        # print(' ')
        # print('p7[aux:aux + n])=', p7[aux:aux + n])
        # print(' ')
        # print('p7[aux + n:aux + n + nel7[i + 1]])=',p7[aux + n:aux + n + nel7[i + 1]])
        # print(' ')

        # guarda a posicao dos vertices nos respectivos vetores
        # Stores the positions of the vertices in their respective arrays.
        u_values = [find1(node[0], p7[aux:aux + n]) for node in lis_final]
        # print('u_values =', u_values)
        v_values = [find1(node[1], p7[aux + n:aux + n + nel7[i + 1]]) for node in lis_final]
        # print('v_values =', v_values)

        # compara arestas e contabiliaza os cruzamentos / Compares edges and tallies the intersections.
        # Ony incremental arcs among each others
        for ii in range(len(lis_uni)):
            u1 = u_values[ii]
            v1 = v_values[ii]
            for jj in range(ii + 1, len(lis_final)):
                u2 = u_values[jj]
                v2 = v_values[jj]
                if (u1 < u2 and v1 > v2) or (u1 > u2 and v1 < v2):
                    cont_sa += 1

    return cont_sa


# u_ e v_ Arrays with the edge positions in the graph.
# u1 outro no - Identification to avoid duplicate counts.
# (u0,v0) Analyzed edge - c_jo Stores intersections
def comp_1(u_, v_, u0, v0, u1, c_j):
    for kk1 in np.arange(np.size(u_)):
        # to avoid counting this/these intersection(s) twice
        if u_[kk1] != u1:
            if (u0 < u_[kk1] and v0 > v_[kk1]) or (u0 > u_[kk1] and v0 < v_[kk1]):
                c_j += 1
    return c_j


def comp_2(u_, v_, u0, v0, c_j):
    for kk1 in np.arange(np.size(u_)):
        if (u0 < u_[kk1] and v0 > v_[kk1]) or (u0 > u_[kk1] and v0 < v_[kk1]):
            c_j += 1
    return c_j


def comp_3(u_, v_, u0, v0, v1, c_j):
    for kk1 in np.arange(np.size(u_)):
        # to avoid counting this/these intersection(s) twice
        if v_[kk1] != v1:
            if (u0 < u_[kk1] and v0 > v_[kk1]) or (u0 > u_[kk1] and v0 < v_[kk1]):
                c_j += 1
    return c_j


def f_swap(v, va, vb, aux, n):
    a = find1(va, v[aux:aux + n])
    b = find1(vb, v[aux:aux + n])
    v[aux + a], v[aux + b] = v[aux + b], v[aux + a]
    return v


# busca local parte 1 / local search part 1
def swap_func(grafos8, nl8, nel8, p8, t_cro, ini_cro):
    best_cost = t_cro + ini_cro
    best_cost_ini = best_cost
    improvement = True

    while improvement:
        improvement = False

        # Percorre camadas / Sweep layers
        for i in np.arange(nl8):
            n = nel8[i]
            aux = np.sum(nel8[0:i]) if i > 0 else 0

            """
            obs.: a lista de nos incrementais por layer deve ter cardinalidade >=2 para que a funcao swap
            faça sentido, uma vez que a função troca a posição entre dois nós incrementais. No caso em que a 
            lista possuir zero ou um elemento, deve-se passar para o proximo layer. 

            Note: The list of incremental nodes per layer must have a cardinality of at least 2 for the swap function 
            to make sense, as the function exchanges the position between two incremental nodes. In the case where the 
            list has zero or one element, you should proceed to the next layer.                             
            """

            # lista de nos incrementais - CONSIDERA POSICAO CORRENTE
            # List of incremental nodes - CONSIDERING CURRENT POSITION.
            n_inc = np.array([nodo for nodo, atri in grafos8[i].nodes(data='id') if atri == 0])
            # guarda as posicoes dos nos incrementais / Stores the positions of the incremental nodes
            ord_inc = np.array([find1(nod1, p8[aux:aux + n]) for nod1 in n_inc])
            # lista com nos incrementais na ordem em que aparecem no layer
            # List of incremental nodes in the order they appear in the layer.
            no_inc = np.array(p8[aux + np.sort(ord_inc)])

            # caso 'no_inc' vazio ou possuir um unico no incremental - ir para proximo layer
            # If 'no_inc' is empty or contains only a single incremental node, move to the next layer
            if np.size(no_inc) <= 1:
                continue
            else:
                for j in no_inc:
                    best_swap = -1
                    novo_inc = np.array([element for element in no_inc if element != j])

                    for w in novo_inc:
                        # contabiliza quantos cruzamentos do total sao originados a partir dos nos j e w
                        # Counts how many intersections in total originate from nodes j and w.
                        # no encontra-se no primeiro layer / Node is in the first layer

                        if i == 0:
                            c_ja = 0
                            c_ka = 0
                            dat_f = np.array(grafos8[i].edges())

                            # guarda a posicao em p dos nos extremos das arestas (head and tail)
                            # Stores the position in 'p' of the endpoints of the edges (head and tail)
                            u_val = np.array([find1(node[0], p8[aux:aux + n]) for node in dat_f])
                            v_val = np.array([find1(node[1], p8[aux + n:aux + n + nel8[i + 1]]) for node in dat_f])

                            # guarda vizinhos do no / keep neighbors of node

                            '''
                            Importante verificar que se no nao possui vizinhos, entao ele nao possui arestas
                            saindo dele. Neste caso, ele e responsavel por zero dos cruzamentos.

                            It's important to verify that if a node doesn't have neighbors, then it doesn't have edges 
                            leaving from it. In this case, it contributes zero to the intersections.
                            '''

                            nei_j = list(grafos8[i].neighbors(j))
                            nei_k = list(grafos8[i].neighbors(w))

                            u1 = find1(j, p8[aux:aux + n])
                            u2 = find1(w, p8[aux:aux + n])

                            if nei_j:
                                for k1 in nei_j:
                                    v1 = find1(k1, p8[aux + n:aux + n + nel8[i + 1]])
                                    c_ja = comp_1(u_val, v_val, u1, v1, u2, c_ja)

                            if nei_k:
                                for k2 in nei_k:
                                    v2 = find1(k2, p8[aux + n:aux + n + nel8[i + 1]])
                                    c_ka = comp_2(u_val, v_val, u2, v2, c_ka)

                            cont_a = c_ja + c_ka

                            # swap
                            p8 = f_swap(p8, j, w, aux, n)

                            # recalcula cruzamento depois do swap / recalculate cross after swap
                            c_jd = 0
                            c_kd = 0

                            # ATUALIZA u_val / UPDATE u_val
                            u_val = np.array([find1(node[0], p8[aux:aux + n]) for node in dat_f])

                            u1_ = find1(j, p8[aux:aux + n])
                            u2_ = find1(w, p8[aux:aux + n])

                            # garantir que nos possuem vizinhos / Ensure that nodes have neighbors
                            if nei_j:
                                for k1 in nei_j:
                                    v1_ = find1(k1, p8[aux + n:aux + n + nel8[i + 1]])
                                    c_jd = comp_1(u_val, v_val, u1_, v1_, u2_, c_jd)

                            if nei_k:
                                for k2 in nei_k:
                                    v2_ = find1(k2, p8[aux + n:aux + n + nel8[i + 1]])
                                    c_kd = comp_2(u_val, v_val, u2_, v2_, c_kd)

                            cont_d = c_jd + c_kd

                            cro_new = best_cost_ini - cont_a + cont_d

                            if cro_new < best_cost:
                                best_cost = cro_new
                                best_swap = w

                            # desfaz permutacao / undo permutation
                            p8 = f_swap(p8, w, j, aux, n)

                        # no esta no utlimo layer / Node is in the last layer
                        if i == nl8 - 1:
                            c_ja = 0
                            c_ka = 0

                            # grafos8[i-1].edges()
                            dat_f = np.array(grafos8[i - 1].edges())

                            # guarda a posicao dos nos extremos das arestas em p8
                            # Stores the positions of the endpoints of the edges in p8.
                            u_val = np.array([find1(node[0], p8[aux - nel8[i - 1]:aux]) for node in dat_f])
                            v_val = np.array([find1(node[1], p8[aux:aux + n]) for node in dat_f])

                            # guarda os vizinhos do no / keep neighbors of node
                            nei_j = list(grafos8[i - 1].predecessors(j))
                            nei_k = list(grafos8[i - 1].predecessors(w))
                            v1 = find1(j, p8[aux:aux + n])
                            v2 = find1(w, p8[aux:aux + n])

                            if nei_j:
                                for k1 in nei_j:
                                    u1 = find1(k1, p8[aux - nel8[i - 1]:aux])
                                    c_ja = comp_3(u_val, v_val, u1, v1, v2, c_ja)

                            if nei_k:
                                for k2 in nei_k:
                                    u2 = find1(k2, p8[aux - nel8[i - 1]:aux])
                                    c_ka = comp_2(u_val, v_val, u2, v2, c_ka)

                            cont_a = c_ja + c_ka

                            # swap
                            p8 = f_swap(p8, j, w, aux, n)

                            # recalcula cruzamento depois do swap / recalculate cross after swap ---------------
                            c_jd = 0
                            c_kd = 0

                            # ATUALIZA v_val / UPDATE v_val
                            v_val = np.array([find1(node[1], p8[aux:aux + n]) for node in dat_f])

                            v1_ = find1(j, p8[aux:aux + n])
                            v2_ = find1(w, p8[aux:aux + n])

                            if nei_j:
                                for k1 in nei_j:
                                    u1_ = find1(k1, p8[aux - nel8[i - 1]:aux])
                                    c_jd = comp_3(u_val, v_val, u1_, v1_, v2_, c_jd)

                            if nei_k:
                                for k2 in nei_k:
                                    u2_ = find1(k2, p8[aux - nel8[i - 1]:aux])
                                    c_kd = comp_2(u_val, v_val, u2_, v2_, c_kd)

                            cont_d = c_jd + c_kd

                            cro_new = best_cost_ini - cont_a + cont_d

                            if cro_new < best_cost:
                                best_cost = cro_new
                                best_swap = w

                            # desfaz permutacao / undo permutation
                            p8 = f_swap(p8, w, j, aux, n)

                        # layer central  / Middle layer
                        elif (i != 0) and (i != nl8 - 1):

                            # depois / next
                            c_af_j = 0
                            c_af_k = 0
                            # aretas do proximo layer / edges next layer
                            dat_aft = np.array(grafos8[i].edges())

                            # guarda a posicao em p dos nos extremos dos arcos (head and tail)
                            # Stores the positions of the endpoints of the arcs in p.
                            u_val = np.array([find1(node[0], p8[aux:aux + n]) for node in dat_aft])
                            v_val = np.array([find1(node[1], p8[aux + n:aux + n + nel8[i + 1]]) for node in dat_aft])

                            # guarda vizinhos / keep neighbors of node
                            nei_ja = list(grafos8[i].neighbors(j))
                            nei_ka = list(grafos8[i].neighbors(w))

                            u1_aft = find1(j, p8[aux:aux + n])
                            u2_aft = find1(w, p8[aux:aux + n])

                            if nei_ja:
                                for k1 in nei_ja:
                                    v1_aft = find1(k1, p8[aux + n:aux + n + nel8[i + 1]])
                                    c_af_j = comp_1(u_val, v_val, u1_aft, v1_aft, u2_aft, c_af_j)

                            if nei_ka:
                                for k2 in nei_ka:
                                    v2_aft = find1(k2, p8[aux + n:aux + n + nel8[i + 1]])
                                    c_af_k = comp_2(u_val, v_val, u2_aft, v2_aft, c_af_k)

                            cont_aft = c_af_j + c_af_k

                            # anterior / previous
                            c_bef_j = 0
                            c_bef_k = 0
                            # arestas layer anterior / edges previous layer
                            dat_bef = np.array(grafos8[i - 1].edges())

                            # guarda a posicao em p dos nos extremos dos arcos
                            # Stores the positions of the endpoints of the arcs in p.
                            u_val_b = np.array([find1(node[0], p8[aux - nel8[i - 1]:aux]) for node in dat_bef])
                            v_val_b = np.array([find1(node[1], p8[aux:aux + n]) for node in dat_bef])

                            # guarda vizinhos do no / keep neighbors of node
                            nei_jb = list(grafos8[i - 1].predecessors(j))
                            nei_kb = list(grafos8[i - 1].predecessors(w))
                            v1_bef = find1(j, p8[aux:aux + n])
                            v2_bef = find1(w, p8[aux:aux + n])

                            if nei_jb:
                                for k1_ in nei_jb:
                                    u1_bef = find1(k1_, p8[aux - nel8[i - 1]:aux])
                                    c_bef_j = comp_3(u_val_b, v_val_b, u1_bef, v1_bef, v2_bef, c_bef_j)

                            if nei_kb:
                                for k2_ in nei_kb:
                                    u2_bef = find1(k2_, p8[aux - nel8[i - 1]:aux])
                                    c_bef_k = comp_2(u_val_b, v_val_b, u2_bef, v2_bef, c_bef_k)

                            cont_bef = c_bef_j + c_bef_k

                            cont_a = cont_aft + cont_bef

                            # swap
                            p8 = f_swap(p8, j, w, aux, n)

                            # recalcula cruzamento depois do swap - atualiza v_val ou u-val
                            # recalculate cross after swap - update v_val ou u_val
                            # proximo / next
                            c_af_j = 0
                            c_af_k = 0

                            # guarda a posicao em p dos nos extremos dos arcos (head and tail)
                            # Stores the positions of the endpoints of the arcs in p
                            u_val = np.array([find1(node[0], p8[aux:aux + n]) for node in dat_aft])
                            # v_val = np.array([find1(node[1], p8[aux + n:aux + n + nel8[i + 1]]) for node in dat_aft])

                            u1_aft_ = find1(j, p8[aux:aux + n])
                            u2_aft_ = find1(w, p8[aux:aux + n])

                            if nei_ja:
                                for k1_a in nei_ja:
                                    v1_aft_ = find1(k1_a, p8[aux + n:aux + n + nel8[i + 1]])
                                    c_af_j = comp_1(u_val, v_val, u1_aft_, v1_aft_, u2_aft_, c_af_j)

                            if nei_ka:
                                for k2_a in nei_ka:
                                    v2_aft_ = find1(k2_a, p8[aux + n:aux + n + nel8[i + 1]])
                                    c_af_k = comp_2(u_val, v_val, u2_aft_, v2_aft_, c_af_k)

                            cont_aft = c_af_j + c_af_k

                            # previous
                            c_bef_j = 0
                            c_bef_k = 0

                            # guarda a posicao em p dos nos extremos dos arcos
                            # Stores the positions of the endpoints of the arcs in p
                            # u_val_b = np.array([find1(node[0], p8[aux - nel8[i - 1]:aux]) for node in dat_bef])
                            v_val_b = np.array([find1(node[1], p8[aux:aux + n]) for node in dat_bef])

                            v1_bef_ = find1(j, p8[aux:aux + n])
                            v2_bef_ = find1(w, p8[aux:aux + n])

                            if nei_jb:
                                for k1__a in nei_jb:
                                    u1_bef_ = find1(k1__a, p8[aux - nel8[i - 1]:aux])
                                    c_bef_j = comp_3(u_val_b, v_val_b, u1_bef_, v1_bef_, v2_bef_, c_bef_j)

                            if nei_kb:
                                for k2__a in nei_kb:
                                    u2_bef_ = find1(k2__a, p8[aux - nel8[i - 1]:aux])
                                    c_bef_k = comp_2(u_val_b, v_val_b, u2_bef_, v2_bef_, c_bef_k)

                            cont_bef = c_bef_j + c_bef_k

                            cont_d = cont_bef + cont_aft

                            cro_new = best_cost_ini - cont_a + cont_d

                            if cro_new < best_cost:
                                best_cost = cro_new
                                best_swap = w
                            # desfaz permutacao / undo permutation
                            p8 = f_swap(p8, w, j, aux, n)

                    if best_swap != -1:
                        p8 = f_swap(p8, j, best_swap, aux, n)
                        best_cost_ini = best_cost
                        improvement = True

    return p8, best_cost


def ins_f(p6, best_pos6, ini_pos6, aux6, no6):
    # Cris copia de p6 / Create a copy of p6 to prevent unwanted modifications to the original array
    temp_p = np.copy(p6)
    # Remove no da lista / Remove the node from the list
    temp_p = np.delete(temp_p, aux6 + ini_pos6)
    # Posicao onde o no sera inserido / Position where the node element will be inserted
    insert_position = best_pos6 + aux6
    # insercao do elemento na posicao / Insert the new element at the calculated position and update p
    up_p = np.insert(temp_p, insert_position, no6)

    return up_p


def down_func(grafos_i, nl_i, nel_i, p_, no_i, k_i, best_pos, ini_pos, best_cost, i, aux, n, ini_cr_i):

    p_i = np.copy(p_)
    best_cost_ini = best_cost

    # primeiro layer / first layer
    if i == 0:
        for j in np.arange(ini_pos - 1, -1, -1):

            # layer atual / p Current layer
            p_t = p_i[aux:aux + n]
            # proximo layer / p next layer
            p_af = p_i[aux + n:aux + n + nel_i[i + 1]]

            # indicador de factibilidade / Feasibility indicator
            ind_i = 0
            # primeiro no anterior ao no em questao / first before node
            no_b = p_t[j]

            # no original / original node
            if grafos_i[i].nodes[no_b]['id'] != 0:
                # posicao original do no / original position of node
                pos_no_b = grafos_i[i].nodes[no_b]['pos_i']
                # verifica a condicao de factibilidade / Checks feasibility condition
                if (j + 1 >= max(0, pos_no_b - k_i)) and (j + 1 <= min(pos_no_b + k_i, n - 1)):
                    ind_i = 1
            # no incremental / incremental node
            else:
                ind_i = 1

            # Tentativa de permutacao / Attempt at permutation application--------
            if ind_i == 1:
                # verificar o numero de cruzamentos antes da mudanca / verify the number of cross before change
                cont_s1a = 0
                cont_s2a = 0
                gra_i = np.array(grafos_i[i].edges())

                # guarda as posicoes dos nos extremos das arestas
                # Store the positions at the endpoints of the edges in variable p. (head and tail)
                u_pos = np.array([find1(node[0], p_t) for node in gra_i])
                v_pos = np.array([find1(node[1], p_af) for node in gra_i])

                # Guarda os vizinhso do no / keep neighbors of node
                nei_i = list(grafos_i[i].neighbors(no_i))
                nei_b = list(grafos_i[i].neighbors(no_b))
                u1 = find1(no_i, p_t)
                u2 = find1(no_b, p_t)

                if nei_i:
                    for k1 in nei_i:
                        v1 = find1(k1, p_af)
                        cont_s1a = comp_1(u_pos, v_pos, u1, v1, u2, cont_s1a)

                if nei_b:
                    for k2 in nei_b:
                        v2 = find1(k2, p_af)
                        cont_s2a = comp_2(u_pos, v_pos, u2, v2, cont_s2a)

                cont_a = cont_s1a + cont_s2a

                # swap
                p_i = f_swap(p_i, no_i, no_b, aux, n)

                # recalcula cruzamentos depois do swap / recalculate cross after swap ------------

                cont_s1d = 0
                cont_s2d = 0

                # atualiza / Update
                u_pos = np.array([find1(node[0], p_i[aux:aux + n]) for node in gra_i])

                u1 = find1(no_i, p_i[aux:aux + n])
                u2 = find1(no_b, p_i[aux:aux + n])

                if nei_i:
                    for k1 in nei_i:
                        v1 = find1(k1, p_af)
                        cont_s1d = comp_1(u_pos, v_pos, u1, v1, u2, cont_s1d)

                if nei_b:
                    for k2 in nei_b:
                        v2 = find1(k2, p_af)
                        cont_s2d = comp_2(u_pos, v_pos, u2, v2, cont_s2d)

                cont_d = cont_s1d + cont_s2d

                cro_new = best_cost_ini - cont_a + cont_d

                if 0 <= cro_new < best_cost:
                    best_cost = cro_new
                    best_pos = j

                best_cost_ini = cro_new

            elif ind_i == 0:
                # quebra restricao / breaches restriction
                p_ = ins_f(p_, best_pos, ini_pos, aux, no_i)
                best_cost = cro_parc_final(grafos_i, nl_i, nel_i, p_) + ini_cr_i
                return p_, best_pos, best_cost

    # ultima camada / last layer
    if i == nl_i - 1:
        for j in np.arange(ini_pos - 1, -1, -1):

            # camada atual / p current layer
            p_t = p_i[aux:aux + n]
            # camada anterior / p previous layer
            p_be = p_i[aux - nel_i[i - 1]:aux]

            # indicador de factibilidade / Feasibility indicator
            ind_i = 0
            # primeiro no antes do no em questao / first before node
            no_b = p_t[j]

            # no original / original node
            if grafos_i[i].nodes[no_b]['id'] != 0:
                # posicao do no original / original node position
                pos_no_b = grafos_i[i].nodes[no_b]['pos_i']
                # verifica restricoes / Check feasibility condition
                if (j + 1 >= max(0, pos_no_b - k_i)) and (j + 1 <= min(pos_no_b + k_i, n - 1)):
                    ind_i = 1
            # no incremental / incremental node
            else:
                ind_i = 1

            # Tentativa de permutacao / Attempt at permutation application--------
            if ind_i == 1:

                # verifica numero de cruzamentos antes da mudanca / verify the number of cross before change
                cont_s1a = 0
                cont_s2a = 0
                gra_i = np.array(grafos_i[i - 1].edges())

                # guarda posicoes dos nos extremos das arestas
                # Store the positions at the endpoints of the edges in variable p (head and tail)
                u_pos = np.array([find1(node[0], p_be) for node in gra_i])
                v_pos = np.array([find1(node[1], p_t) for node in gra_i])

                # guarda vizinhos do no / keep neighbors of node
                nei_i = list(grafos_i[i - 1].predecessors(no_i))
                nei_b = list(grafos_i[i - 1].predecessors(no_b))
                v1 = find1(no_i, p_t)
                v2 = find1(no_b, p_t)

                if nei_i:
                    for k1 in nei_i:
                        u1 = find1(k1, p_be)
                        cont_s1a = comp_3(u_pos, v_pos, u1, v1, v2, cont_s1a)

                if nei_b:
                    for k2 in nei_b:
                        u2 = find1(k2, p_be)
                        cont_s2a = comp_2(u_pos, v_pos, u2, v2, cont_s2a)

                cont_a = cont_s1a + cont_s2a

                # swap
                p_i = f_swap(p_i, no_i, no_b, aux, n)

                # recalcula cruzamentos depois do swap / recalculate cross after swap ------------

                cont_s1d = 0
                cont_s2d = 0

                # atualiza / Update
                v_pos = np.array([find1(node[1], p_i[aux:aux + n]) for node in gra_i])

                v1 = find1(no_i, p_i[aux:aux + n])
                v2 = find1(no_b, p_i[aux:aux + n])

                if nei_i:
                    for k1 in nei_i:
                        u1 = find1(k1, p_be)
                        cont_s1d = comp_3(u_pos, v_pos, u1, v1, v2, cont_s1d)

                if nei_b:
                    for k2 in nei_b:
                        u2 = find1(k2, p_be)
                        cont_s2d = comp_2(u_pos, v_pos, u2, v2, cont_s2d)

                cont_d = cont_s1d + cont_s2d

                cro_new = best_cost_ini - cont_a + cont_d
                if 0 <= cro_new < best_cost:
                    best_cost = cro_new
                    best_pos = j

                best_cost_ini = cro_new

            else:
                # quebra restricao / breaches restriction
                p_ = ins_f(p_, best_pos, ini_pos, aux, no_i)
                best_cost = cro_parc_final(grafos_i, nl_i, nel_i, p_) + ini_cr_i
                return p_, best_pos, best_cost

    # layer central / middle layer
    elif (i != 0) and (i != nl_i - 1):

        for j in np.arange(ini_pos - 1, -1, -1):

            # layer atual / p current layer
            p_t = p_i[aux:aux + n]
            # proximo layer / p next layer
            p_af = p_i[aux + n:aux + n + nel_i[i + 1]]
            # layer anterior / p previous layer
            p_be = p_i[aux - nel_i[i - 1]:aux]

            # indicador de factibilidade / Feasibility indicator
            ind_i = 0
            # primeiro no antes do no em questao / first before node
            no_b = p_t[j]

            # no original / original node
            if grafos_i[i].nodes[no_b]['id'] != 0:
                # posicao do no original / position original node
                pos_no_b = grafos_i[i].nodes[no_b]['pos_i']
                # verifica restricao / Check feasibility condition
                if (j + 1 >= max(0, pos_no_b - k_i)) and (j + 1 <= min(pos_no_b + k_i, n - 1)):
                    ind_i = 1
            # no incremental / incremental node
            else:
                ind_i = 1

            # tentativa de permutacao / Attempt at permutation application --------
            if ind_i == 1:

                # proxima camada / next Layer
                # verifica o numero de cruzamentos antes da mudanca / verify the number of cross before change
                cont_s1ad = 0
                cont_s2ad = 0
                gra_id = np.array(grafos_i[i].edges())

                # guarda as posicoes dos nos extremos das arestas
                # Store the positions at the endpoints of the edges in variable p (head and tail)
                u_posd = np.array([find1(node[0], p_t) for node in gra_id])
                v_posd = np.array([find1(node[1], p_af) for node in gra_id])

                # guarda os vizinhos dos nos / keep neighbors of node
                nei_id = list(grafos_i[i].neighbors(no_i))
                nei_bd = list(grafos_i[i].neighbors(no_b))
                u1d = find1(no_i, p_t)
                u2d = find1(no_b, p_t)

                if nei_id:
                    for k1 in nei_id:
                        v1d = find1(k1, p_af)
                        cont_s1ad = comp_1(u_posd, v_posd, u1d, v1d, u2d, cont_s1ad)

                if nei_bd:
                    for k2 in nei_bd:
                        v2d = find1(k2, p_af)
                        cont_s2ad = comp_2(u_posd, v_posd, u2d, v2d, cont_s2ad)

                cont_ad = cont_s1ad + cont_s2ad

                # camada anterior / previous Layer
                # verifica o numero de cruzamentos antes da mudanca / verify the number of cross before change
                cont_s1ab = 0
                cont_s2ab = 0
                gra_ib = np.array(grafos_i[i - 1].edges())

                # guarda as posicoes dos nos extremos das arestas
                # Store the positions at the endpoints of the edges in variable p (head and tail)
                u_posb = np.array([find1(node[0], p_be) for node in gra_ib])
                v_posb = np.array([find1(node[1], p_t) for node in gra_ib])

                # guarda os vizinhos do no / keep neighbors of node
                nei_ib = list(grafos_i[i - 1].predecessors(no_i))
                nei_bb = list(grafos_i[i - 1].predecessors(no_b))
                v1b = find1(no_i, p_t)
                v2b = find1(no_b, p_t)

                if nei_ib:
                    for k1 in nei_ib:
                        u1b = find1(k1, p_be)
                        cont_s1ab = comp_3(u_posb, v_posb, u1b, v1b, v2b, cont_s1ab)

                if nei_bb:
                    for k2 in nei_bb:
                        u2b = find1(k2, p_be)
                        cont_s2ab = comp_2(u_posb, v_posb, u2b, v2b, cont_s2ab)

                cont_ab = cont_s1ab + cont_s2ab

                cont_a = cont_ad + cont_ab

                # swap
                p_i = f_swap(p_i, no_i, no_b, aux, n)

                # recalcula cruZamentos depois do swap / recalculate cross after swap -------------------------

                # proxima camada / next Layer
                # verifica o numero de cruzamentos antes da mudanca / verify the number of cross before change
                cont_s1ad = 0
                cont_s2ad = 0

                # guarda as posicoes dos nos extremos das arestas
                # Store the positions at the endpoints of the edges in variable p (head and tail)
                u_posd = np.array([find1(node[0], p_i[aux:aux + n]) for node in gra_id])

                u1d = find1(no_i, p_i[aux:aux + n])
                u2d = find1(no_b, p_i[aux:aux + n])

                if nei_id:
                    for k1 in nei_id:
                        v1d = find1(k1, p_af)
                        cont_s1ad = comp_1(u_posd, v_posd, u1d, v1d, u2d, cont_s1ad)

                if nei_bd:
                    for k2 in nei_bd:
                        v2d = find1(k2, p_af)
                        cont_s2ad = comp_2(u_posd, v_posd, u2d, v2d, cont_s2ad)

                cont_ad = cont_s1ad + cont_s2ad

                # camada anterior / previous Layer
                # verifica o numero de cruzamentos antes da mudanca / verify the number of cross before change
                cont_s1ab = 0
                cont_s2ab = 0

                # guarda as posicoes dos nos extremos das arestas
                # Store the positions at the endpoints of the edges in variable p (head and tail)
                v_posb = np.array([find1(node[1], p_i[aux:aux + n]) for node in gra_ib])

                v1b = find1(no_i, p_i[aux:aux + n])
                v2b = find1(no_b, p_i[aux:aux + n])

                if nei_ib:
                    for k1 in nei_ib:
                        u1b = find1(k1, p_be)
                        cont_s1ab = comp_3(u_posb, v_posb, u1b, v1b, v2b, cont_s1ab)

                if nei_bb:
                    for k2 in nei_bb:
                        u2b = find1(k2, p_be)
                        cont_s2ab = comp_2(u_posb, v_posb, u2b, v2b, cont_s2ab)

                cont_ab = cont_s1ab + cont_s2ab

                cont_d = cont_ad + cont_ab

                cro_new = best_cost_ini - cont_a + cont_d
                if 0 <= cro_new < best_cost:
                    best_cost = cro_new
                    best_pos = j

                best_cost_ini = cro_new

            else:
                # quebra restricao / breaches restriction
                p_ = ins_f(p_, best_pos, ini_pos, aux, no_i)
                best_cost = cro_parc_final(grafos_i, nl_i, nel_i, p_) + ini_cr_i
                return p_, best_pos, best_cost

    p_ = ins_f(p_, best_pos, ini_pos, aux, no_i)
    best_cost = cro_parc_final(grafos_i, nl_i, nel_i, p_) + ini_cr_i

    return p_, best_pos, best_cost


def up_func(grafos_i, nl_i, nel_i, p_, no_i, k_i, best_pos, ini_pos, best_cost, i, aux, n, ini_cr_i):
    p_i = np.copy(p_)

    best_cost_ini = best_cost

    # primeira camada / first layer
    if i == 0:
        for j in np.arange(ini_pos + 1, n, 1):

            # camada atual / p current layer
            p_t = p_i[aux:aux + n]
            # proxima camada / p next layer posterior
            p_af = p_i[aux + n:aux + n + nel_i[i + 1]]

            # indicador de factibilidade / Feasibility indicator
            ind_i = 0
            # primeiro no anterior ao no atual / first before node
            no_b = p_t[j]

            # no original / original node
            if grafos_i[i].nodes[no_b]['id'] != 0:
                # posicao do no original / position original node
                pos_no_b = grafos_i[i].nodes[no_b]['pos_i']
                # verifica restricao / Check feasibility condition
                if (j - 1 >= max(0, pos_no_b - k_i)) and (j - 1 <= min(pos_no_b + k_i, n - 1)):
                    ind_i = 1
            # no incremental / incremental node
            else:
                ind_i = 1

            # tentativa de permutacao / Attempt at permutation application --------
            if ind_i == 1:
                # verifica o numero de cruzamentos antes da mudanca / verify the number of cross before change
                cont_s1a = 0
                cont_s2a = 0
                gra_i = np.array(grafos_i[i].edges())

                # guarda as posicoes dos nos extremos das arestas
                # Store the positions at the endpoints of the edges in variable p (head and tail)
                u_pos = np.array([find1(node[0], p_t) for node in gra_i])
                v_pos = np.array([find1(node[1], p_af) for node in gra_i])

                # guarda vizinhos do no / keep neighbors of node
                nei_i = list(grafos_i[i].neighbors(no_i))
                nei_b = list(grafos_i[i].neighbors(no_b))
                u1 = find1(no_i, p_t)
                u2 = find1(no_b, p_t)

                if nei_i:
                    for k1 in nei_i:
                        v1 = find1(k1, p_af)
                        cont_s1a = comp_1(u_pos, v_pos, u1, v1, u2, cont_s1a)

                if nei_b:
                    for k2 in nei_b:
                        v2 = find1(k2, p_af)
                        cont_s2a = comp_2(u_pos, v_pos, u2, v2, cont_s2a)

                cont_a = cont_s1a + cont_s2a

                # swap
                p_i = f_swap(p_i, no_i, no_b, aux, n)

                # recalcula cruzamentos depois do swap / recalculate cross after swap ------------

                cont_s1d = 0
                cont_s2d = 0

                # atualiza / Update
                u_pos = np.array([find1(node[0], p_i[aux:aux + n]) for node in gra_i])

                u1 = find1(no_i, p_i[aux:aux + n])
                u2 = find1(no_b, p_i[aux:aux + n])

                if nei_i:
                    for k1 in nei_i:
                        v1 = find1(k1, p_af)
                        cont_s1d = comp_1(u_pos, v_pos, u1, v1, u2, cont_s1d)

                if nei_b:
                    for k2 in nei_b:
                        v2 = find1(k2, p_af)
                        cont_s2d = comp_2(u_pos, v_pos, u2, v2, cont_s2d)

                cont_d = cont_s1d + cont_s2d

                cro_new = best_cost_ini - cont_a + cont_d
                if 0 <= cro_new < best_cost:
                    best_cost = cro_new
                    best_pos = j

                best_cost_ini = cro_new

            else:
                # quebra restricao / breaches restriction
                p_ = ins_f(p_, best_pos, ini_pos, aux, no_i)
                best_cost = cro_parc_final(grafos_i, nl_i, nel_i, p_) + ini_cr_i
                return p_, best_pos, best_cost

    # ultima camada / last layer
    if i == nl_i - 1:
        for j in np.arange(ini_pos + 1, n, 1):

            # camada atual / p current layer
            p_t = p_i[aux:aux + n]
            # camada anterior / p previous layer
            p_be = p_i[aux - nel_i[i - 1]:aux]

            # indicador de factibilidade / Feasibility indicator
            ind_i = 0
            # primeiro no antes do no em questao / first before node
            no_b = p_t[j]

            # no original / original node
            if grafos_i[i].nodes[no_b]['id'] != 0:
                # posicao do no orignal / position original node
                pos_no_b = grafos_i[i].nodes[no_b]['pos_i']
                # verifica condicao de factibilidade / Check feasibility condition
                if (j - 1 >= max(0, pos_no_b - k_i)) and (j - 1 <= min(pos_no_b + k_i, n - 1)):
                    ind_i = 1
            # no incremental / incremental node
            else:
                ind_i = 1

            # tentativa de permutacao / Attempt at permutation application --------
            if ind_i == 1:

                # verifica cruzamentos antes da mudanca / verify the number of cross before change
                cont_s1a = 0
                cont_s2a = 0
                gra_i = np.array(grafos_i[i - 1].edges())

                # guarda as posicoes dos nos extremos das arestas
                # Store the positions at the endpoints of the edges in variable p (head and tail)
                u_pos = np.array([find1(node[0], p_be) for node in gra_i])
                v_pos = np.array([find1(node[1], p_t) for node in gra_i])

                # guarda os vizinhos do no / keep neighbors of node
                nei_i = list(grafos_i[i - 1].predecessors(no_i))
                nei_b = list(grafos_i[i - 1].predecessors(no_b))
                v1 = find1(no_i, p_t)
                v2 = find1(no_b, p_t)

                if nei_i:
                    for k1 in nei_i:
                        u1 = find1(k1, p_be)
                        cont_s1a = comp_3(u_pos, v_pos, u1, v1, v2, cont_s1a)

                if nei_b:
                    for k2 in nei_b:
                        u2 = find1(k2, p_be)
                        cont_s2a = comp_2(u_pos, v_pos, u2, v2, cont_s2a)

                cont_a = cont_s1a + cont_s2a

                # swap
                p_i = f_swap(p_i, no_i, no_b, aux, n)

                # recalcula cruzamentos depois do swap / recalculate cross after swap ------------

                cont_s1d = 0
                cont_s2d = 0

                # atualiza / Update
                v_pos = np.array([find1(node[1], p_i[aux:aux + n]) for node in gra_i])

                v1 = find1(no_i, p_i[aux:aux + n])
                v2 = find1(no_b, p_i[aux:aux + n])

                if nei_i:
                    for k1 in nei_i:
                        u1 = find1(k1, p_be)
                        cont_s1d = comp_3(u_pos, v_pos, u1, v1, v2, cont_s1d)

                if nei_b:
                    for k2 in nei_b:
                        u2 = find1(k2, p_be)
                        cont_s2d = comp_2(u_pos, v_pos, u2, v2, cont_s2d)

                cont_d = cont_s1d + cont_s2d

                cro_new = best_cost_ini - cont_a + cont_d
                if 0 <= cro_new < best_cost:
                    best_cost = cro_new
                    best_pos = j

                best_cost_ini = cro_new

            else:
                # quebra restricao / breaches restriction
                p_ = ins_f(p_, best_pos, ini_pos, aux, no_i)
                best_cost = cro_parc_final(grafos_i, nl_i, nel_i, p_) + ini_cr_i
                return p_, best_pos, best_cost

    # camada central / middle layer
    elif (i != 0) and (i != nl_i - 1):

        for j in np.arange(ini_pos + 1, n, 1):

            # camada atual / p current layer
            p_t = p_i[aux:aux + n]
            # proxima camada / p next layer
            p_af = p_i[aux + n:aux + n + nel_i[i + 1]]
            # camada anterior / p previous layer
            p_be = p_i[aux - nel_i[i - 1]:aux]

            # indicador de factibilidade / Feasibility indicator
            ind_i = 0
            # primeiro no anterior ao no em questao / first before node
            no_b = p_t[j]

            # no original / original node
            if grafos_i[i].nodes[no_b]['id'] != 0:
                # posicao do no original / position original node
                pos_no_b = grafos_i[i].nodes[no_b]['pos_i']
                # verifica factibilidade / Check feasibility condition
                if (j - 1 >= max(0, pos_no_b - k_i)) and (j - 1 <= min(pos_no_b + k_i, n - 1)):
                    ind_i = 1
            # no incremental / incremental node
            else:
                ind_i = 1

            # tentativa de permutacao / Attempt at permutation application --------
            if ind_i == 1:

                # proximo layer / next Layer
                # verifica cruzamentos antes da mudanca / verify the number of cross before change
                cont_s1ad = 0
                cont_s2ad = 0
                gra_id = np.array(grafos_i[i].edges())

                # guarda as posicoes dos nos extremos das arestas
                # Store the positions at the endpoints of the edges in variable p (head and tail)
                u_posd = np.array([find1(node[0], p_t) for node in gra_id])
                v_posd = np.array([find1(node[1], p_af) for node in gra_id])

                # guarda vizinhos do no / keep neighbors of node
                nei_id = list(grafos_i[i].neighbors(no_i))
                nei_bd = list(grafos_i[i].neighbors(no_b))
                u1d = find1(no_i, p_t)
                u2d = find1(no_b, p_t)

                if nei_id:
                    for k1 in nei_id:
                        v1d = find1(k1, p_af)
                        cont_s1ad = comp_1(u_posd, v_posd, u1d, v1d, u2d, cont_s1ad)

                if nei_bd:
                    for k2 in nei_bd:
                        v2d = find1(k2, p_af)
                        cont_s2ad = comp_2(u_posd, v_posd, u2d, v2d, cont_s2ad)

                cont_ad = cont_s1ad + cont_s2ad

                # camada anterior / previous Layer
                # verifica cruzamentos antes da mudanca / verify the number of cross before change
                cont_s1ab = 0
                cont_s2ab = 0
                gra_ib = np.array(grafos_i[i - 1].edges())

                # guarda as posicoes dos nos extremos das arestas
                # Store the positions at the endpoints of the edges in variable p. (head and tail)
                u_posb = np.array([find1(node[0], p_be) for node in gra_ib])
                v_posb = np.array([find1(node[1], p_t) for node in gra_ib])

                # guarda vizinhos do no / keep neighbors of node
                nei_ib = list(grafos_i[i - 1].predecessors(no_i))
                nei_bb = list(grafos_i[i - 1].predecessors(no_b))
                v1b = find1(no_i, p_t)
                v2b = find1(no_b, p_t)

                if nei_ib:
                    for k1 in nei_ib:
                        u1b = find1(k1, p_be)
                        cont_s1ab = comp_3(u_posb, v_posb, u1b, v1b, v2b, cont_s1ab)

                if nei_bb:
                    for k2 in nei_bb:
                        u2b = find1(k2, p_be)
                        cont_s2ab = comp_2(u_posb, v_posb, u2b, v2b, cont_s2ab)

                cont_ab = cont_s1ab + cont_s2ab

                cont_a = cont_ad + cont_ab

                # swap
                p_i = f_swap(p_i, no_i, no_b, aux, n)

                # recalcula cruzamenrto depois do swap / recalculate cross after swap -------------------------

                # proxima camada / Next Layer
                # verifica cruzamentos antes da mudanca / verify the number of cross before change
                cont_s1ad = 0
                cont_s2ad = 0

                # guarda as posicoes dos nos extremos das arestas
                # Store the positions at the endpoints of the edges in variable p (head and tail)
                u_posd = np.array([find1(node[0], p_i[aux:aux + n]) for node in gra_id])

                u1d = find1(no_i, p_i[aux:aux + n])
                u2d = find1(no_b, p_i[aux:aux + n])

                if nei_id:
                    for k1 in nei_id:
                        v1d = find1(k1, p_af)
                        cont_s1ad = comp_1(u_posd, v_posd, u1d, v1d, u2d, cont_s1ad)

                if nei_bd:
                    for k2 in nei_bd:
                        v2d = find1(k2, p_af)
                        cont_s2ad = comp_2(u_posd, v_posd, u2d, v2d, cont_s2ad)

                cont_ad = cont_s1ad + cont_s2ad

                # camada anterior / Previous Layer
                # verifica cruzamentos antes do swap / verify the number of cross before change
                cont_s1ab = 0
                cont_s2ab = 0

                # guarda as posicoes dos nos extremos das arestas
                # Store the positions at the endpoints of the edges in variable p (head and tail)
                v_posb = np.array([find1(node[1], p_i[aux:aux + n]) for node in gra_ib])

                v1b = find1(no_i, p_i[aux:aux + n])
                v2b = find1(no_b, p_i[aux:aux + n])

                if nei_ib:
                    for k1 in nei_ib:
                        u1b = find1(k1, p_be)
                        cont_s1ab = comp_3(u_posb, v_posb, u1b, v1b, v2b, cont_s1ab)

                if nei_bb:
                    for k2 in nei_bb:
                        u2b = find1(k2, p_be)
                        cont_s2ab = comp_2(u_posb, v_posb, u2b, v2b, cont_s2ab)

                cont_ab = cont_s1ab + cont_s2ab

                cont_d = cont_ad + cont_ab

                cro_new = best_cost_ini - cont_a + cont_d
                if 0 <= cro_new < best_cost:
                    best_cost = cro_new
                    best_pos = j

                best_cost_ini = cro_new

            else:
                # quebra restricao / breaches restriction
                p_ = ins_f(p_, best_pos, ini_pos, aux, no_i)
                best_cost = cro_parc_final(grafos_i, nl_i, nel_i, p_) + ini_cr_i
                return p_, best_pos, best_cost

    p_ = ins_f(p_, best_pos, ini_pos, aux, no_i)
    best_cost = cro_parc_final(grafos_i, nl_i, nel_i, p_) + ini_cr_i

    return p_, best_pos, best_cost


# Local search part 2
def insertion_func(grafos_if, nl_if, nel_if, cro_if, p_i0, k_if, ini_cr):

    improvement = True

    while improvement:
        b_cost = cro_if
        cro_if_ini = cro_if
        improvement = False

        # percorre camadas / Sweep layers
        for io in np.arange(nl_if):
            n = nel_if[io]
            aux = np.sum(nel_if[0:io]) if io > 0 else 0

            # lista de nos incrementais no layer / List of incremental nodes in the layer.
            no_inc = np.array([nodo for nodo, atri in grafos_if[io].nodes(data='id') if atri == 0])
            # guarda as posicoes dos nos incrementais / Stores the positions of the incremental nodes.
            ord_inc = np.array([find1(j, p_i0[aux:aux + n]) for j in no_inc])
            # lista com nos incrementais na ordem em que aparecem no layer
            # List of incremental nodes in the order they appear in the layer.
            nos_incr = np.array(p_i0[aux + np.sort(ord_inc)])

            # percorre nos incrementais no layer atual / Sweep incremental node in current layer
            for no_i in nos_incr:
                b_pos = find1(no_i, p_i0[aux:aux + n])
                i_pos = b_pos

                # garante que existe ao menos uma posicao anterior ao no atual
                # Ensure that there is at least one position before the current node
                if b_pos != 0:
                    (p_i0, b_pos, b_cost_) = down_func(grafos_if, nl_if, nel_if, p_i0, no_i, k_if, b_pos, i_pos, b_cost,
                                                       io,
                                                       aux, n, ini_cr)

                    cro_if = b_cost_
                    if b_pos == i_pos:
                        # garante que existe ao menos uma posicao posterior ao no atual
                        # Ensure that there is at least one position after the current node.
                        if b_pos < n - 1:
                            (p_i0, b_pos, b_cost) = up_func(grafos_if, nl_if, nel_if, p_i0, no_i, k_if, b_pos, i_pos,
                                                            b_cost_, io, aux, n, ini_cr)
                            cro_if = b_cost

                            # print(' ')
                            # print('best cost second part insertion=', b_cost)
                            # print('verifica result insert firt part=', cro_parc_final(grafos_if, nl_if, nel_if, p_i0)
                            # + ini_cr)
                            # print('------')

        # or cro_if_ini != cro_if
        if cro_if_ini > cro_if:
            improvement = True

    return p_i0, cro_if


# GRASP Constructive Phase
def greedy_rand_construction(grafos_t, nlt, nel_t, pert, clt, kt, alpha_t):
    # Pre_Const_PhaseCb
    d_min0_t, d_min_pos0_t, rcl0_t = pre_const_phase_cb(grafos_t, nlt, nel_t, pert, clt, alpha_t)

    # The call to const_phase_cb function
    per_n = const_phase_cb(grafos_t, nlt, nel_t, pert, clt, kt, alpha_t, d_min0_t, d_min_pos0_t, rcl0_t)

    # Partial crossover accounting using p - No duplicate crossings
    cro_pt = cro_parc_final(grafos_t, nlt, nel_t, per_n)

    return per_n, cro_pt


# improvement grasp
def local_search_f(grafos_w, nlw, nel_w, per_cons_w, cro_pw, ini_cross_w, kw):
    # local search part 1 - swap function - return total cross pos swap
    per_new, cro_tot_new = swap_func(grafos_w, nlw, nel_w, per_cons_w, cro_pw, ini_cross_w)

    # compute crossings
    cro_p2 = cro_parc_final(grafos_w, nlw, nel_w, per_new)
    # partial result
    # arq_hope.write(str(cro_p2 + ini_cross_w) + '\n')

    # local search part 2 - insertion function - final result per_f e cro_f
    per_fw, c_fw = insertion_func(grafos_w, nlw, nel_w, cro_p2 + ini_cross_w, per_new, kw, ini_cross_w)
    cro_fw = cro_parc_final(grafos_w, nlw, nel_w, per_fw) + ini_cross_w

    return per_fw, cro_fw


# grasp
def grasp(grafos_v, nlv, nel_v, perv, clv, kv, alphav, ini_cro_v):
    # Constructive Phase
    per_cons_v, cro_pv = greedy_rand_construction(grafos_v, nlv, nel_v, perv, clv, kv, alphav)

    # local search
    solution, cost = local_search_f(grafos_v, nlv, nel_v, per_cons_v, cro_pv, ini_cro_v, kv)

    return solution, cost, per_cons_v, cro_pv+ini_cro_v


def test_k(graphx, nel_x, nlx):

    # guarda numero minimo de no incremental por layer
    # keep the minimum number of incremental node per layer
    min_x = float('inf')
    graf_x = graphx[0:, 0:1:1]

    # percorre camadas / sweep layers
    for ix in np.arange(nlx):
        n = nel_x[ix]
        aux = np.sum(nel_x[0:ix]) if ix > 0 else 0

        # contabiliza numero de nos incrementais no layer
        # compute the number of incremental nodes in layer
        minx = cont(graf_x[aux:aux+n])
        if minx < min_x:
            min_x = minx

    return min_x


# MAIN()------------------------------------------------------------------------------##

def main():
    #  Reading data
    all_graphs = data_read()
    # total runs 30 - DEFINE
    total_run = int(input("Runs number:"))
    # GRASP Iteration 100 - DEFINE
    it_ma = int(input("GRASP Iteration:"))
    # stop critter 20 - DEFINE
    stop_ = int(input("Stop Critter:"))

    for ru in np.arange(total_run):

        # keep the crossings numbers constructive and final phase
        arq_sdne = open('res_sdne2/Cb_SDNE2_' + str(ru+1) + '.txt', 'a')
        # keep the permutation value
        arq_sdne_p = open('res_sdne2/Cb_SDNE2_p_' + str(ru+1) + '.txt', 'a')
        # keep the permutation value pos constructive phase
        arq_sdne_con_p = open('res_sdne2/Cb_SDNE2_const_p_' + str(ru+1) + '.txt', 'a')
        # keep grasp iterations
        arq_sdne_iter = open('res_sdne2/Cb_SDNE2_iter_' + str(ru+1) + '.txt', 'a')

        # the number of incremental nodes per layer must be greater than or equal to k.
        k_all = [1, 2, 3]
        for k in k_all:

            # Iterating through the instances (arrays) of the all_graphs list
            for ide, (graph, nel, nl) in enumerate(all_graphs):

                # controla núm nós incr >= k / control node number >= k
                t_k = test_k(graph, nel, nl)
                if t_k < k:
                    arq_sdne.write(str(ide) + str(',') + str(k) + str(',') + str(-1) + str(',') + str(-1) + str(',') +
                                   str(-1) + '\n')
                    arq_sdne_p.write(str(ide) + str(',') + str(k) + str(',') + str(-1) + '\n')
                    arq_sdne_con_p.write(str(ide) + str(',') + str(k) + str(',') + str(-1) + '\n')

                else:
                    start_time = time.monotonic()

                    best_solution = None
                    best_cost = float('inf')
                    # result from constructive phase
                    best_solution_con = None
                    best_cost_con = float('inf')

                    # control the stop critter
                    stop_aux = 0

                    #print(node_ids)
                    #print(emb_2d)
                    #print(' ')

                    for ite in np.arange(it_ma):

                        # each iteration time begin
                        start_time_in = time.monotonic()

                        # temp value alpha
                        alpha = random.uniform(0, 1)

                        # UM EMBEDDING POR ITERACAO GRASP / ONE EMBEDDING FOR EACH GRASP ITERATION
                        # embedding construction return nodes ids and embeddings coordinates
                        node_ids, emb_2d = pre_emb_data(graph, nl, nel)

                        # initialize variables
                        cl, ini_cross, grafos_, per = begin_f(graph, nel, nl, node_ids, emb_2d)
                        # compute the euclidian distance and Assigns it to edges graph
                        grafos = ares_atr(grafos_, nl)

                        # call grasp
                        x_, f_x, p_con, cro_con = grasp(grafos, nl, nel, per, cl, k, alpha, ini_cross)

                        stop_aux += 1

                        # seleciona melhor solucao
                        if f_x < best_cost:
                            best_solution = x_
                            best_cost = f_x
                            best_solution_con = p_con
                            best_cost_con = cro_con
                            stop_aux = 0

                        # break loop in para executions
                        if stop_aux == stop_:
                            break

                        # each iteration time end
                        end_time_in = time.monotonic()

                        # id graph, k, alpha, iteration, f(x) constructive phase, f(x) final, time
                        arq_sdne_iter.write(str(ide) + str(',') + str(k) + str(',') + str(round(alpha, 2)) + str(',') +
                                            str(ite + 1) + str(',') + str(best_cost_con) + str(',') + str(best_cost) +
                                            str(',') + str(timedelta(seconds=end_time_in - start_time_in)) + '\n')

                    end_time = time.monotonic()

                    # id graph, k, f(x) constructive phase, f(x) final, time
                    arq_sdne.write(str(ide) + str(',') + str(k) + str(',') + str(best_cost_con) + str(',') +
                                   str(best_cost) + str(',') + str(timedelta(seconds=end_time - start_time)) + '\n')
                    arq_sdne_p.write(str(ide) + str(',') + str(k) + str(',') + str(best_solution) + '\n')
                    arq_sdne_con_p.write(str(ide) + str(',') + str(k) + str(',') + str(best_solution_con) + '\n')

                # break

        arq_sdne.close()
        arq_sdne_p.close()
        arq_sdne_con_p.close()
        arq_sdne_iter.close()


# Executa o programa
if __name__ == "__main__":
    main()
