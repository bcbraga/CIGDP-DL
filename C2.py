"""
Este código é uma implementação feita por Bruna Cristina Braga Charytitsch do método GRASP2: C2 + Local Search proposto
por Napoletano et al. (2019) <https://www.sciencedirect.com/science/article/abs/pii/S0377221718308701>. A implementação
foi feita em linguagem Python para posterior comparação deste método da literatura com aqueles propostos pela
mesma autora.

Date: [26/06/2024]

This code is an implementation by Bruna Cristina Braga Charytitsch of the GRASP2: C2 + Local Search method proposed by
Napoletano et al. (2019) <https://www.sciencedirect.com/science/article/abs/pii/S0377221718308701>. The implementation
was done in Python for subsequent comparison of this literature method with those proposed by the same author.

Date: [06/26/2024]
"""


# BIBLIOTECAS/LIBRARIES

import numpy as np
from pandas import DataFrame
import time
from datetime import timedelta
import networkx as nx
import collections as col
import random


# BEGIN DATA READING:--------------------------------------------------------------##
def data_read():
    graph_id = []
    file_path = 'data/ids'

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
    g = nx.parse_adjlist(a2, nodetype=int, create_using=nx.DiGraph)

    # remove auxiliary -1
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


# Update p array
def update_p(no_, vet_, pos_no_):
    vet_[pos_no_ - 1], vet_[pos_no_] = no_, vet_[pos_no_ - 1]
    return vet_


def atualizap(no_, vet_, pos_no_):
    vet_[pos_no_ - 1], vet_[pos_no_] = no_, vet_[pos_no_ - 1]
    return vet_


# Function FC to calculate the total of arcs crossings in the graph.
# Returns initial crossings and an array with one digraph per layer.
def fc(graf, la, ne, inc):
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

        # Preparing an array with la-1 graphs (1 per layer).
        gr[i] = gra2

    n0 = ne[la - 1]
    aux0 = np.sum(ne[0:la - 1])
    gra3 = nx.DiGraph()

    for i3 in graf[aux0:aux0 + n0:1, 1]:

        a = find1(i3, graf[aux0:aux0 + n0:1, 1])
        # original nodes
        if graf[aux0 + a, 0] == 1:
            gra3.add_node(i3, id=1, cam=la - 1, pos_i=a)
        # incremental nodes
        if graf[aux0 + a, 0] == 0:
            gra3.add_node(i3, id=0, cam=la - 1, pos_i=a)

    # graph form the last layer
    gr[la - 1] = gra3

    return s_cross, gr


# variable initialization
def begin_f(graph_, nel_, nl_):

    # construction of candidate list / Array with incremental nodes - rows represent layers
    cl_ = np.array(cand_list(graph_[0:, 0:2:1], nl_, nel_))

    # Returns the initial crossing and nl-1 digraphs (NetworkX)
    ini_cross_, grafos_ = fc(graph_, nl_, nel_, cl_)

    # permutation vector
    per_ = np.array(perm2(graph_[0:, 0:2:1]))

    return cl_, ini_cross_, grafos_, per_


def test_k(graphx, nel_x, nlx):

    # keep the minimum number of incremental node per layer
    min_x = float('inf')
    graf_x = graphx[0:, 0:1:1]

    # sweep layers
    for ix in np.arange(nlx):
        n = nel_x[ix]
        aux = np.sum(nel_x[0:ix]) if ix > 0 else 0

        # compute the number of incremental nodes in layer
        minx = cont(graf_x[aux:aux+n])
        if minx < min_x:
            min_x = minx

    return min_x


# Deg function
def deg_func(g0, nl0, nel0, per0, no0, l0):

    n = nel0[l0]
    aux = np.sum(nel0[0:l0]) if l0 > 0 else 0

    # keep the degree number
    tot_grad = 0
    #  keep neighbor node position
    som_pos_viz = 0
    # New position
    new_pos = None

    # first layer
    if l0 == 0:

        dep = list(g0[l0].neighbors(no0))
        tem_e = any(ele in dep for ele in per0[aux+n:aux+n+nel0[l0+1]])

        # if the node does not have neighbors
        if (not dep) or (dep and not tem_e):
            # First available position at the end
            new_pos = list(per0[aux:aux + n]).index(-1)

        else:
            # stores the already inserted neighbor in p
            temp_p = per0[aux+n:aux+n+nel0[l0+1]]
            neigh = np.array([dep[i] for i in np.arange(len(dep)) if (dep[i] in temp_p)])

            for viz in neigh:
                tot_grad += 1
                som_pos_viz += find1(viz, temp_p)

            new_pos = som_pos_viz // tot_grad

            # Case in which the average of the neighbors positions exceeds the size of the current layer
            if new_pos >= n:
                new_pos = n-1

    # last layer
    if l0 == nl0 - 1:

        ant = list(g0[l0 - 1].predecessors(no0))
        tem_e = any(ele in ant for ele in per0[aux - nel0[l0 - 1]:aux])

        # if the node does not have neighbors
        if (not ant) or (ant and not tem_e):
            # First available position at the end
            new_pos = list(per0[aux:aux + n]).index(-1)

        else:
            # stores the already inserted neighbor in p
            temp_p = per0[aux - nel0[l0 - 1]:aux]
            neigh = np.array([ant[i] for i in np.arange(len(ant)) if (ant[i] in temp_p)])

            for viz in neigh:
                tot_grad += 1
                som_pos_viz += find1(viz, temp_p)

            new_pos = som_pos_viz // tot_grad

            # Case in which the average of the neighbors positions exceeds the size of the current layer
            if new_pos >= n:
                new_pos = n - 1

    # middle layer
    if (l0 > 0) and (l0 < nl0 - 1):

        dep = list(g0[l0].neighbors(no0))
        tem_e_d = any(ele in dep for ele in per0[aux+n:aux+n+nel0[l0+1]])

        ant = list(g0[l0 - 1].predecessors(no0))
        tem_e_a = any(ele in ant for ele in per0[aux - nel0[l0 - 1]:aux])

        # they do not have any subsequent or preceding neighbors
        if dep == [] and ant == []:
            # First available position at the end
            new_pos = list(per0[aux:aux + n]).index(-1)

        # they do not have neighbors inserted in either the previous or the subsequent layer
        elif(tem_e_d is False) and (tem_e_a is False):
            # First available position at the end
            new_pos = list(per0[aux:aux + n]).index(-1)

        else:

            # it only has neighbors in the next layer
            if (tem_e_a is False) and (tem_e_d is True):

                temp_p_next = per0[aux + n:aux + n + nel0[l0 + 1]]
                neigh_next = np.array([dep[i] for i in np.arange(len(dep)) if (dep[i] in temp_p_next)])

                for viz in neigh_next:
                    tot_grad += 1
                    som_pos_viz += find1(viz, temp_p_next)

                new_pos = som_pos_viz // tot_grad

                # Case in which the average of the neighbors positions exceeds the size of the current layer
                if new_pos >= n:
                    new_pos = n - 1

            # it only has predecessors in the previous layer
            if (tem_e_a is True) and (tem_e_d is False):

                temp_p_pre = per0[aux - nel0[l0 - 1]:aux]
                neigh_pre = np.array([ant[i] for i in np.arange(len(ant)) if (ant[i] in temp_p_pre)])

                for viz in neigh_pre:
                    tot_grad += 1
                    som_pos_viz += find1(viz, temp_p_pre)

                new_pos = som_pos_viz // tot_grad

                # Case in which the average of the neighbors positions exceeds the size of the current layer
                if new_pos >= n:
                    new_pos = n - 1

            # it only has neighbors in both previous and next layer
            if (tem_e_a is True) and (tem_e_d is True):

                # keep the degree number
                tot_grad_next = 0
                #  keep neighbor node position
                som_pos_viz_next = 0

                temp_p_next = per0[aux + n:aux + n + nel0[l0 + 1]]
                neigh_next = np.array([dep[i] for i in np.arange(len(dep)) if (dep[i] in temp_p_next)])

                for viz in neigh_next:
                    tot_grad_next += 1
                    som_pos_viz_next += find1(viz, temp_p_next)

                # keep the degree number
                tot_grad_pre = 0
                #  keep neighbor node position
                som_pos_viz_pre = 0

                temp_p_pre = per0[aux - nel0[l0 - 1]:aux]
                neigh_pre = np.array([ant[i] for i in np.arange(len(ant)) if (ant[i] in temp_p_pre)])

                for viz in neigh_pre:
                    tot_grad_pre += 1
                    som_pos_viz_pre += find1(viz, temp_p_pre)

                tot_grad = tot_grad_pre + tot_grad_next

                new_pos = (som_pos_viz_pre // (2 * tot_grad_pre)) + (som_pos_viz_next // (2 * tot_grad_next))

                # Case in which the average of the neighbors positions exceeds the size of the current layer
                if new_pos >= n:
                    new_pos = n - 1

    return tot_grad, new_pos


# Check if the position can be filled with 'node'
def check_position(grafos4, k4, layer4, no4, cont4, aux4, i4):
    node_info = grafos4[layer4].nodes[aux4[i4 - 1]]
    id_value = node_info['id']

    # incremental node
    if id_value == 0:
        aux4 = update_p(no4, aux4, i4)
        cont4 += 1
    # original node
    elif id_value == 1:
        pos_i = node_info['pos_i']
        lower_bound = max(0, pos_i - k4)
        upper_bound = min(pos_i + k4, aux4.shape[0] - 1)

        if lower_bound <= i4 <= upper_bound:
            aux4 = update_p(no4, aux4, i4)
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


def pre_const_phase_c2(grafos1, nl1, nel1, per1, cl1, alp1):
    # Generates matrices (layers x elements) - rows = layers - columns = incremental nodes (blank spaces = -1)

    # (1) Computing g(v)
    a_deg = (-1) * np.ones((nl1, cl1.shape[1]), dtype=int)
    a_best_pos = (-1) * np.ones((nl1, cl1.shape[1]), dtype=int)

    for i in np.arange(nl1):
        for j in np.arange(cl1.shape[1]):
            pos = cl1[i, j]
            if pos != -1:
                a_deg[i, j], a_best_pos[i, j] = deg_func(grafos1, nl1, nel1, per1, pos, i)

    # computing global max
    max_g = np.max(a_deg[a_deg != -1])

    # Compute the threshold value only once."
    threshold = alp1 * max_g
    # Create a matrix of -1 values with the same dimensions as a_deg
    rcl_ = np.full_like(a_deg, -1)
    # Identify the indices where the condition is met
    valid_indices = np.where((a_deg >= threshold) & (a_deg != -1))
    # Assign the values from cl1 where the condition is met
    rcl_[valid_indices] = cl1[valid_indices]

    return a_deg, a_best_pos, rcl_


def pre_const_phase_c2_2(grafos1_, nl1_, nel1_, per1_, cl1_, alp1_, la1_, a_deg1, a_best_pos_):

    for i in la1_:
        for j in np.arange(cl1_.shape[1]):
            pos = cl1_[i, j]
            if pos != -1:
                a_deg1[i, j], a_best_pos_[i, j] = deg_func(grafos1_, nl1_, nel1_, per1_, pos, i)

    # computing global max
    masked_array = np.ma.masked_equal(a_deg1, -1)
    max_g = np.max(masked_array)

    # Compute the threshold value only once."
    threshold = alp1_ * max_g
    # Create a matrix of -1 values with the same dimensions as a_deg
    rcl_1 = np.full_like(a_deg1, -1)
    # Identify the indices where the condition is met
    valid_indices = np.where((a_deg1 >= threshold) & (a_deg1 != -1))
    # Assign the values from cl1 where the condition is met
    rcl_1[valid_indices] = cl1_[valid_indices]

    return a_deg1, a_best_pos_, rcl_1, per1_


# Construction of an initial solution
def const_phase_cb(grafos2, nl2, nel2, per2, cl2, k2, alp2, deg2, best_pos_i2, rcl2):

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
        best_pos_ = g_vq(grafos2, nel2, per2, k2, x, cl2[x][y], best_pos_i2[x][y])
        # I update the position where the node will be inserted
        best_pos_i2[x][y] = best_pos_

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
        insert_position = best_pos_i2[x][y] + aux
        # Insert the new element at the calculated position and update p
        up_p = np.insert(temp_p, insert_position, cl2[x][y])
        # -----

        # update cd
        cl2[x][y] = -1
        # updates:
        best_pos_i2[x][y] = -1
        deg2[x][y] = -1

        # recompute g and tau, rebuild RCL
        if x == 0:
            (deg2, best_pos_i2, rcl2, per2) = pre_const_phase_c2_2(grafos2, nl2, nel2, up_p, cl2, alp2, [x, x + 1],
                                                                   deg2, best_pos_i2)

        # Last Layer (update previous and current layers)
        if x == nl2 - 1:
            (deg2, best_pos_i2, rcl2, per2) = pre_const_phase_c2_2(grafos2, nl2, nel2, up_p, cl2, alp2, [x - 1, x],
                                                                   deg2, best_pos_i2)

        # middle layer (update layer previous and current layer)
        if (x != 0) and (x != nl2 - 1):
            (deg2, best_pos_i2, rcl2, per2) = pre_const_phase_c2_2(grafos2, nl2, nel2, up_p, cl2, alp2,
                                                                   [x - 1, x, x + 1], deg2, best_pos_i2)

    return per2


def comp_cro(grafos7_, p1_7, p2_7, u7, v7, cro7):
    for u, v in grafos7_.edges():
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


# constructive phase grasp
def greedy_rand_construction(grafos_t, nlt, nelt, pert, clt, kt, alphat):

    # The first call to pre_const_phase_c3  function
    (deg_, best, rcl_f) = pre_const_phase_c2(grafos_t, nlt, nelt, pert, clt, alphat)

    # The call to const_phase_c3 function
    (per_const) = const_phase_cb(grafos_t, nlt, nelt, pert, clt, kt, alphat, deg_, best, rcl_f)

    # Partial intersection tally using p - without duplicate intersections.
    cro_pt = cro_parc_final(grafos_t, nlt, nelt, per_const)

    return per_const, cro_pt


# improvement grasp
def local_search_f(grafosw, nlw, nelw, per_consw, cro_pw, ini_crossw, kw):

    # local search part 1 - swap function - return total cros pos swap
    per_new, cro_tot_new = swap_func(grafosw, nlw, nelw, per_consw, cro_pw, ini_crossw)
    # print('Total cross new=', cro_tot_new)
    # print('P new=', per_new)

    # local search part 2 - insertion function
    # compute cros pos swap
    cro_p2 = cro_parc_final(grafosw, nlw, nelw, per_new)
    # Final result  per_f e cro_f
    per_fw, c_fw = insertion_func(grafosw, nlw, nelw, cro_p2 + ini_crossw, per_new, kw, ini_crossw)
    cro_fw = cro_parc_final(grafosw, nlw, nelw, per_fw) + ini_crossw
    # print(ini_cross + cro_p, cro_f + ini_cross)
    # print('compara=', c_f, cro_f + ini_cross)

    return per_fw, cro_fw


# grasp
def grasp(grafos_v, nlv, nel_v, perv, clv, kv, alphav, ini_cro_v):

    # Constructive Phase
    per_cons_v, cro_pv = greedy_rand_construction(grafos_v, nlv, nel_v, perv, clv, kv, alphav)

    # local search
    solution, cost = local_search_f(grafos_v, nlv, nel_v, per_cons_v, cro_pv, ini_cro_v, kv)

    return solution, cost, per_cons_v, cro_pv+ini_cro_v


# MAIN()------------------------------------------------------------------------------##

def main():
    # Reading the data
    all_graphs = data_read()
    # total runs 30 - DEFINE
    total_run = int(input("Runs number:"))
    # GRASP Iteration 100 - DEFINE
    it_ma = int(input("GRASP Iteration:"))
    # stop critter 20 - DEFINE
    stop_ = int(input("Stop Critter:"))

    for ru in np.arange(total_run):

        # keep the crossings numbers constructive and final phase
        arq_c2 = open('res_c2/C2_' + str(ru+1) + '.txt', 'a')
        # keep the permutation value
        arq_c2_p = open('res_c2/C2_p_' + str(ru+1) + '.txt', 'a')
        # keep the permutation value pos constructive phase
        arq_c2_con_p = open('res_c2/C2_const_p_' + str(ru+1) + '.txt', 'a')
        # keep grasp iterations
        arq_c2_iter = open('res_c2/C2_iter_' + str(ru+1) + '.txt', 'a')

        # the number of incremental nodes per layer must be greater than or equal to k.
        k_all = [1, 2, 3]
        for k in k_all:

            # Iterating through the instances (arrays) of the all_graphs list
            for ind_, (graph, nel, nl) in enumerate(all_graphs):

                # control node number >= k
                t_k = test_k(graph, nel, nl)
                if t_k < k:
                    arq_c2.write(str(ind_) + str(',') + str(k) + str(',') + str(-1) + str(',') + str(-1) + str(',')
                                 + str(-1) + '\n')
                    arq_c2_p.write(str(ind_) + str(',') + str(k) + str(',') + str(-1) + '\n')
                    arq_c2_con_p.write(str(ind_) + str(',') + str(k) + str(',') + str(-1) + '\n')

                else:

                    start_time = time.monotonic()

                    best_solution = None
                    best_cost = float('inf')
                    # result from constructive phase
                    best_solution_con = None
                    best_cost_con = float('inf')

                    # control the stop critter
                    stop_aux = 0

                    for ite in np.arange(it_ma):

                        # each iteration time begin
                        start_time_in = time.monotonic()

                        # temp value alpha
                        alpha = random.uniform(0, 1)

                        # variable initialization
                        cl, ini_cross, grafos, per = begin_f(graph, nel, nl)
                        # call grasp
                        x_, f_x, p_con, cro_con = grasp(grafos, nl, nel, per, cl, k, alpha, ini_cross)

                        stop_aux += 1

                        # select the best solution
                        if f_x < best_cost:
                            best_solution = x_
                            best_cost = f_x
                            best_solution_con = p_con
                            best_cost_con = cro_con
                            stop_aux = 0

                        # break loop in stop_ executions
                        if stop_aux == stop_:
                            break

                        # each iteration time end
                        end_time_in = time.monotonic()

                        # id graph, k, alpha, iteration, f(x) constructive phase, f(x) final, time
                        arq_c2_iter.write(str(ind_) + str(',') + str(k) + str(',') + str(round(alpha, 2)) + str(',') +
                                          str(ite + 1) + str(',') + str(best_cost_con) + str(',') + str(best_cost)
                                          + str(',') + str(timedelta(seconds=end_time_in - start_time_in)) + '\n')

                    end_time = time.monotonic()

                    # id graph, k, f(x) constructive phase, f(x) final, time
                    arq_c2.write(
                        str(ind_) + str(',') + str(k) + str(',') + str(best_cost_con) + str(',') + str(best_cost)
                        + str(',') + str(timedelta(seconds=end_time - start_time)) + '\n')
                    arq_c2_p.write(str(ind_) + str(',') + str(k) + str(',') + str(best_solution) + '\n')
                    arq_c2_con_p.write(str(ind_) + str(',') + str(k) + str(',') + str(best_solution_con) + '\n')

        arq_c2.close()
        arq_c2_p.close()
        arq_c2_con_p.close()
        arq_c2_iter.close()


# Run program
if __name__ == "__main__":
    main()
