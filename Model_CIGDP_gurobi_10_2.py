# BIBLIOTECAS/LIBRARIES

import numpy as np
from pandas import DataFrame
# import pandas as pd

from time import perf_counter
from datetime import timedelta
import networkx as nx
from itertools import permutations  # combinations
# import collections as col
# import random
# from pulp import *

import gurobipy as gp
from gurobipy import GRB

from pathlib import Path


# GRB_LICENSE_FILE = 'gurobi.lic'


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

            # Example of access to attributes
            # print('BRUNA - id=', gra.nodes[5]['id'], 'cam=', gra.nodes[5]['cam'], 'pos_i=', gra.nodes[5]['pos_i'])

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
        minx = cont(graf_x[aux:aux + n])
        if minx < min_x:
            min_x = minx

    return min_x


# MAIN()------------------------------------------------------------------------------##

def main():

    # Ensure output folder exists
    Path('res_model').mkdir(parents=True, exist_ok=True)

    # Reading the data
    all_graphs = data_read()

    # keep the crossings numbers constructive and final phase
    arq_model = open('res_model/model.txt', 'a')
    # keep the permutation value
    arq_model_p = open('res_model/model_p_.txt', 'a')

    # the number of incremental nodes per layer must be greater than or equal to k.
    k_all = [1, 2, 3]
    # k_all = [3]
    for k in k_all:

        # Iterating through the instances (arrays) of the all_graphs list
        for ind_, (graph, nel, nl) in enumerate(all_graphs):

            # control node number >= k
            t_k = test_k(graph, nel, nl)
            if t_k < k:
                arq_model.write(str(ind_) + str(',') + str(k) + str(',') + str(-1) + str(',') + str(-1) + str(',') +
                                str(-1) + str(',') + str(-1) + '\n')
                arq_model_p.write(str(ind_) + str(',') + str(k) + str(',') + str(-1) + '\n')

            else:

                # each iteration time begin

                # Input data - parameters
                cl, ini_cross, grafos, per = begin_f(graph, nel, nl)

                start_time = perf_counter()

                # keep edges of graph per layer - arcos[layer][head node][tail node]
                arcos = []
                # Keep original nodes per layer
                ori_nodes = []
                # keep the position of original nodes per layer
                pos_ori_nodes = []
                # all nodes
                all_nodes = []
                # position of all nodes
                pos_all_nodes = []

                # Sweep Layers - 0 a nl-2
                for i1 in np.arange(nl - 1):

                    list_aux1 = []
                    list_aux2 = []
                    list_aux3 = []

                    arcos.append(list(grafos[i1].edges()))

                    if i1 < nl - 2:
                        for u, v in list(grafos[i1].edges()):

                            if (grafos[i1].nodes[u]['id'] == 1) and (u not in list_aux1):
                                list_aux1.append(u)
                                list_aux2.append(grafos[i1].nodes[u]['pos_i'])

                            if u not in list_aux3:
                                list_aux3.append(u)

                        ori_nodes.append(list_aux1)
                        pos_ori_nodes.append(list_aux2)
                        all_nodes.append(list_aux3)

                    elif i1 == nl - 2:

                        for u, v in list(grafos[i1].edges()):
                            if (grafos[i1].nodes[u]['id'] == 1) and (u not in list_aux1):
                                list_aux1.append(u)
                                list_aux2.append(grafos[i1].nodes[u]['pos_i'])

                            if u not in list_aux3:
                                list_aux3.append(u)

                        ori_nodes.append(list_aux1)
                        pos_ori_nodes.append(list_aux2)
                        all_nodes.append(list_aux3)

                        list_aux1 = []
                        list_aux2 = []
                        list_aux3 = []

                        for u, v in list(grafos[i1].edges()):
                            if (grafos[i1 + 1].nodes[v]['id'] == 1) and (v not in list_aux1):
                                list_aux1.append(v)
                                list_aux2.append(grafos[i1 + 1].nodes[v]['pos_i'])

                            if v not in list_aux3:
                                list_aux3.append(v)

                        ori_nodes.append(list_aux1)
                        pos_ori_nodes.append(list_aux2)
                        all_nodes.append(list_aux3)

                # Build pos_all_nodes - all nodes position
                for cot1 in np.arange(nl):
                    list1 = []
                    for cot2 in all_nodes[cot1]:
                        list1.append(grafos[cot1].nodes[cot2]['pos_i'])
                    pos_all_nodes.append(list1)

                # Combines arcs per layer------------------------------------------------------
                arc_unic = []
                for i2 in np.arange(nl - 1):
                    arc_aux = set()
                    # sweep arcs and ads single combinations in set
                    # for (vi, vj), (vk, vl) in permutations(arcos_, 2):  # permutations combinations
                    for (vi, vj), (vk, vl) in permutations(list(grafos[i2].edges()), 2):  # permutations combinations
                        arc_aux.add(((vi, vj), (vk, vl)))
                        # print(((vi, vj), (vk, vl)))

                    arc_unic.append(arc_aux)

                # -------------------------------

                # initial permutation vector - (layer, node): position
                p0_vars = {(ca, all_nodes[ca][i9]): pos_all_nodes[ca][i9] for ca in np.arange(nl) for i9 in
                           np.arange(len(all_nodes[ca]))}

                # -----------------------------------------------------------------------------------------------------

                # Create a new model
                cigdp = gp.Model(name="GIGDP")

                # Create variables

                # binaries decision variables / Use list comprehension
                x_vars = {(ca, va, vb): cigdp.addVar(lb=0, ub=1, obj=0, vtype=gp.GRB.BINARY,
                                                     name=f'x_{ca}_{va}_{vb}', column=None)
                          for ca in np.arange(nl)
                          for va in all_nodes[ca]
                          for vb in all_nodes[ca] if va != vb}

                # final permutation vector
                p_vars = {(ca, vi): cigdp.addVar(lb=0, ub=len(all_nodes[ca]) - 1, obj=0, vtype=gp.GRB.INTEGER,
                                                 name=f'p_{ca}_{vi}', column=None)
                          for ca in np.arange(nl)
                          for vi in all_nodes[ca]}

                c_vars = {}
                for ca in range(nl - 1):
                    for (vi, vj), (vk, vl) in (arc_unic[ca]):
                        if vi != vk and vj != vl:
                            c_vars[(ca, vi, vj, vk, vl)] = cigdp.addVar(0, 1, 0, vtype=gp.GRB.BINARY,
                                                                        name="c_{}_{}_{}_{}_{}"
                                                                        .format(ca, vi, vj, vk, vl), column=None)


                # Set objective
                objective = gp.quicksum([c_vars[ca, vi, vj, vk, vl] for ca in range(nl - 1) for (vi, vj), (vk, vl)
                                         in arc_unic[ca] if vi != vk and vj != vl])

                # Add constraints

                # BEGIN RESTRICTS (7) and (8) - OK
                for p_ in np.arange(nl - 1):
                    for (i_, j_), (k_, l_) in arc_unic[p_]:
                        if i_ != k_ and j_ != l_:

                            if p0_vars[p_, i_] < p0_vars[p_, k_] and p0_vars[p_ + 1, j_] < p0_vars[p_ + 1, l_]:
                                # restriction (7)
                                cigdp.addConstr((-1) * c_vars[p_, i_, j_, k_, l_] <= x_vars[p_ + 1, j_, l_] -
                                                x_vars[p_, i_, k_], name='7a')
                                cigdp.addConstr(x_vars[p_ + 1, j_, l_] - x_vars[p_, i_, k_] <=
                                                c_vars[p_, i_, j_, k_, l_], name='7b')

                            if p0_vars[p_, i_] < p0_vars[p_, k_] and p0_vars[p_ + 1, j_] > p0_vars[p_ + 1, l_]:
                                # restriction (8)
                                cigdp.addConstr(1 - c_vars[p_, i_, j_, k_, l_] <= x_vars[p_ + 1, l_, j_] +
                                                x_vars[p_, i_, k_], name='8a')
                                cigdp.addConstr(x_vars[p_ + 1, l_, j_] + x_vars[p_, i_, k_] <= 1 +
                                                c_vars[p_, i_, j_, k_, l_], name='8b')

                # BEGIN RESTRICTS (9) - OK

                for p_ in np.arange(nl):
                    for i_ in all_nodes[p_]:
                        for j_ in all_nodes[p_]:
                            for k_ in all_nodes[p_]:
                                if i_ != j_ and j_ != k_ and i_ != k_:

                                    if p0_vars[p_, i_] < p0_vars[p_, j_] < p0_vars[p_, k_]:
                                        cigdp.addConstr(x_vars[p_, i_, j_] + x_vars[p_, j_, k_] -
                                                        x_vars[p_, i_, k_] <= 1, name='9a')
                                        cigdp.addConstr(0 <= x_vars[p_, i_, j_] + x_vars[p_, j_, k_] -
                                                        x_vars[p_, i_, k_], name='9b')

                # BEGIN RESTRICTS (10) - OK

                for p_ in np.arange(nl):
                    for i_ in all_nodes[p_]:
                        for j_ in all_nodes[p_]:
                            if i_ != j_:

                                if p0_vars[p_, i_] < p0_vars[p_, j_]:
                                    cigdp.addConstr(x_vars[p_, i_, j_] + x_vars[p_, j_, i_] == 1, name='10')

                # BEGIN RESTRICTS (11) - OK
                for p_ in np.arange(nl):
                    for i_ in ori_nodes[p_]:
                        for j_ in ori_nodes[p_]:
                            if i_ != j_:
                                if p0_vars[p_, i_] < p0_vars[p_, j_]:
                                    cigdp.addConstr(x_vars[p_, i_, j_] == 1, name='11a')
                                elif p0_vars[p_, i_] > p0_vars[p_, j_]:
                                    cigdp.addConstr(x_vars[p_, i_, j_] == 0, name='11b')

                # BEGIN RESTRICTS (12) and (13) - OK
                for p_ in np.arange(nl):
                    for i_ in ori_nodes[p_]:
                        cigdp.addConstr(p_vars[p_, i_] >= max(0, p0_vars[p_, i_] - k), name='12')
                        cigdp.addConstr(p_vars[p_, i_] <= min(p0_vars[p_, i_] + k, len(all_nodes[p_]) - 1),
                                        name='13')

                # RESTRICAO EXTRA
                for p_ in np.arange(nl):
                    for i_ in all_nodes[p_]:
                        cont_ = 0
                        for j_ in all_nodes[p_]:
                            if i_ != j_:
                                cont_ += x_vars[p_, i_, j_]

                        cigdp.addConstr(p_vars[p_, i_] == len(all_nodes[p_]) - 1 - cont_, 'Extra')

                # for minimization
                cigdp.ModelSense = gp.GRB.MINIMIZE
                cigdp.setObjective(objective)

                # Optimize model
                cigdp.optimize()
                # for v in cigdp.getVars():
                #    print('%s %g' % (v.VarName, v.X))

                if cigdp.Status == gp.GRB.OPTIMAL:

                    # pos processing -----------------------------------------------

                    perm = list()
                    for ca in np.arange(nl):
                        aux_perm = list(np.zeros(len(all_nodes[ca])))
                        for vi in (all_nodes[ca]):
                            aux_perm[round(p_vars[ca, vi].X)] = vi
                        perm.append(aux_perm)

                    print('Obj: %g' % cigdp.ObjVal)
                    print(perm)
                    end_time = perf_counter()
                    print('time=', timedelta(seconds=end_time - start_time), 'tempo=', end_time - start_time)

                    # id graph, k, f(x) Objective function, Problem Status, time, time
                    arq_model.write(
                        str(ind_) + str(',') + str(k) + str(',') + str(int(cigdp.ObjVal)) + str(',') +
                        str(GRB.OPTIMAL) + str(',') + str(timedelta(seconds=end_time - start_time)) +
                        str(',') + str(end_time - start_time) + '\n')
                    # permutation vector p - graph index and vector p
                    arq_model_p.write(str(ind_) + str(',') + str(k) + str(',') + str(perm) + '\n')

                else:
                    end_time = perf_counter()
                    # id graph, k, f(x) Objective function, Problem Status, time, time
                    arq_model.write(
                        str(ind_) + str(',') + str(k) + str(',') + str(int(cigdp.ObjVal)) + str(',') +
                        str(cigdp.Status) + str(',') + str(timedelta(seconds=end_time - start_time)) +
                        str(',') + str(end_time - start_time) + str('ERROR') + '\n')
                    # permutation vector p - graph index and vector p
                    arq_model_p.write(str(ind_) + str(',') + str(k) + str(',') + str(cigdp.getAttrArray('X', p_vars)) +
                                      str('ERROR') + '\n')

                # -----------------------------

            # stop
            # break

    arq_model.close()
    arq_model_p.close()


# Run program
if __name__ == "__main__":
    main()
