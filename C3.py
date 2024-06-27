"""
Este código é uma implementação feita por Bruna Cristina Braga Charytitsch do método GRASP3: C3 + Local Search proposto
por Napoletano et al. (2019) <https://www.sciencedirect.com/science/article/abs/pii/S0377221718308701>. A implementação
foi feita em linguagem Python para posterior comparação deste método da literatura com aqueles propostos pela
mesma autora.

Date: [26/06/2024]

This code is an implementation by Bruna Cristina Braga Charytitsch of the GRASP3: C3 + Local Search method proposed by
Napoletano et al. (2019) <https://www.sciencedirect.com/science/article/abs/pii/S0377221718308701>. The implementation
was done in Python for subsequent comparison of this literature method with those proposed by the same author.

Date: [06/26/2024]
"""


# BIBLIOTECAS/LIBRARIES

import numpy as np
import pandas as pd
import time
from datetime import timedelta
import networkx as nx
import collections as col
import random


# INICIO LEITURA DOS DADOS:---------------------------------------------------------##
# BEGIN DATA READING:--------------------------------------------------------------##
def data_read():
    graph_id = []
    file_path = 'data/ids'

    """
    A lista graph_id contém as linhas do arquivo ids, sem os caracteres de nova linha.
    with open(...) para garantir que o arquivo seja fechado corretamente após a leitura.

    The list graph_id contains the lines from the file "ids", without newline characters.
    Using "with open(...)" to ensure that the file is properly closed after reading.
    """
    with open(file_path, 'r') as file:
        for line in file:
            graph_id.append(line.strip())
    # tot_graphs = len(graph_id)

    """
    Explicações:
    A lista all_graphs é usada para armazenar os dados de todos os arquivos. Cada elemento da lista contém três itens: 
    o array graph, o array nel e o inteiro nl.Utilizamos o gerenciador de contexto with para abrir e fechar os arquivos, 
    evitando deixar arquivos abertos desnecessariamente. Durante a leitura do arquivo, já convertemos as strings em 
    inteiros usando map, tornando o processo mais eficiente. Após o loop, temos a lista all_graphs que contém as 
    informações de todos os arquivos em formato de arrays NumPy.

    Explanations:
    The list "all_graphs" is used to store data from all the files. Each element of the list contains three items:
    the "graph" array, the "nel" array, and the integer "nl". We use the "with" context manager to open and close the 
    files, avoiding leaving files open unnecessarily. During the file reading, we convert the strings to integers using
    "map", making the process more efficient. After the loop, we have the "all_graphs" list containing information from
    all the files in the form of NumPy arrays.
    """

    # Lista para armazenar os dados dos arquivos
    # List to store the data from the files.
    all_graphs = []

    for temp in graph_id:
        file_path = 'data/' + str.strip(temp) + '.txt'

        with open(file_path, 'r') as file:
            # Removendo as linhas em branco do início e do final e convertendo para inteiros
            # Removing leading and trailing blank lines and converting to integers.
            lines = [list(map(int, line.split())) for line in file if line.strip()]

        # Verifica se existem dados suficientes no arquivo
        # Checking if there is enough data in the file.
        if len(lines) >= 3:
            nl = lines[0][0]
            nel = lines[1]
            graph = lines[2:]

            df = pd.DataFrame(graph).fillna(-1).astype(int)
            df_a = df.to_numpy(dtype=int)

            all_graphs.append((df_a, np.array(nel), nl))

    # all_graphs[0][0] - Retun graph / retorna grafo
    # all_graphs[0][1] - Return nel / retorna nel
    # all_graphs[0][0] - Return nl / retorna nl
    return all_graphs


# FIM LEITURA DOS DADOS:---------------------------------------------------------##
# END DATA READING:--------------------------------------------------------------##

# FUNÇÕES--------------------------------------------##
# FUNCTIONS------------------------------------------##

# Retorna a quantidade de nós incrementais a partir da primeira coluna do vetor.
# Return quantity of incremental nodes from the first column of the vector
def cont(v1):
    return np.count_nonzero(v1 == 0)


# Retorna a quantidade de nós incrementais a partir da primeira coluna do vetor após a inserção dos nós incrementais.
# Return quantity of incremental nodes from the first column of the vector pos insertion incremental nodes
def cont2(v2):
    return np.count_nonzero(v2 == -1)


# Aplicar a máscara e retornar os elementos na linha - vizinhança.
# Apply the mask and return the elements on the line - neighborhood
def masc(v3):
    return v3[v3 != -1]


# Retorna o índice após verificar se não é um nó incremental.
# Return index after verifying if it is not an incremental node
def find1(el, x):
    return np.where(el == x)[0][0]


# Identifica cruzamento.
# Identify cross
def cross(u_, ul_, v_, vl_):
    return int((u_ < ul_ and v_ > vl_) or (u_ > ul_ and v_ < vl_))


# Dado o grafo no formato inicial, gera uma lista de adjascência e retorna um Digrafo (NetworkX)
# Given the graph in the initial format, generates an adjacency list and returns a DiGraph (NetworkX).
def make_graph(v4):
    a1 = v4.tolist()
    a2 = []
    for i in np.arange(len(a1)):
        t1 = str(a1[i]).strip('[]')
        t2 = t1.replace(',', ' ')
        a2.append(t2)
    g = nx.parse_adjlist(a2, nodetype=int, create_using=nx.DiGraph)

    # remove auxiliar -1 / remove auxiliary -1
    if -1 in g:
        g.remove_node(-1)

    return g


# Retorna a lista de candidatos
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


# Retorna vetor de permutacao com nós originais em suas posições iniciais e os espaços restantes com -1
# Returns a permutation vector with original nodes in their initial positions and the remaining spaces with -1.
def perm2(v5):
    return list(map(lambda x, y: x * y if y != 0 else -1, v5[0:, 1], v5[0:, 0]))


def atualizap(no_, vet_, pos_no_):
    vet_[pos_no_ - 1], vet_[pos_no_] = no_, vet_[pos_no_ - 1]
    return vet_


# Função FC para calcular o número total de cruzamentos de arcos no grafo
# Function FC to calculate the total of arcs crossings in the graph.
# Retorna cruzamento inicial e array com um digrafo por layer
# Returns initial crossings and an array with one digrafo per layer.
def fc(graf, la, ne, inc):
    s_cross = 0
    # Guarda os Digrafos gerados
    # keep Digraphs generated
    gr = np.empty(la, dtype=object)

    for i in np.arange(la - 1):
        n = ne[i]
        aux = np.sum(ne[0:i]) if i > 0 else 0

        # contabiliza total de nos incrementais
        # compute the total of incremental nodes
        i_n = cont(graf[aux:aux + n, 0])
        gra = make_graph(graf[aux:aux + n - i_n:1, 1:])

        # atualiza lista de arcos eliminando nós sucessores ainda não inseridos
        # Update arc list by removing successor nodes that have not been inserted yet.
        for uu, vv in list(gra.edges()):
            # atributos nos originais (somente nos origens de arcos no layer atual)
            # attributes original nodes (only in the source nodes of arcs in the current layer)
            gra.nodes[uu]['id'] = 1
            gra.nodes[uu]['cam'] = i
            gra.nodes[uu]['pos_i'] = find1(uu, graf[aux:aux + n:1, 1])

            # Exemplo de acesso aos atributos
            # Example of access to attributes
            # print('BRUNA - id=', gra.nodes[5]['id'], 'cam=', gra.nodes[5]['cam'], 'pos_i=', gra.nodes[5]['pos_i'])

        # grafo auxiliar / auxiliary graph
        gra2 = gra.copy()

        # removendo vizinhos incrementais
        # removing incremental neighbors
        for u, v in list(gra.edges()):
            if v in inc[i + 1]:
                gra.remove_edge(u, v)

        for i1 in graf[aux + n - i_n:aux + n:1, 1]:
            for i2 in masc(graf[aux + find1(i1, graf[aux:aux + n:1, 1]), 2:]):
                gra2.add_edge(i1, i2)

            # atributos nos incrementais (origem) / incremental nodes atributes (head)
            gra2.nodes[i1]['id'] = 0
            gra2.nodes[i1]['cam'] = i
            gra2.nodes[i1]['pos_i'] = find1(i1, graf[aux:aux + n:1, 1])

        # Guarda os nós origens e destinos de cada arco
        # keep the head and tail nodes of each arc
        arst = np.array(gra.edges)

        # Guarda a posição inicial dos nós origens e destinos de cada arco
        # Keep the initial position of the head and tail nodes of each arc.
        a_pos = np.zeros((arst.shape[0], 2), dtype=int)

        for c1 in np.arange(arst.shape[0]):
            a_pos[c1, 0] = find1(arst[c1, 0], graf[aux:aux + n:1, 1:2:1])
            a_pos[c1, 1] = find1(arst[c1, 1], graf[aux + n:aux + n + ne[i + 1]:1, 1:2:1])

        # Calculando Cruzamentos
        # Computing crossings
        for k1 in np.arange(a_pos.shape[0] - 1):
            for j in np.arange(k1 + 1, a_pos.shape[0], 1):
                s_cross += cross(a_pos[k1][0], a_pos[j][0], a_pos[k1][1], a_pos[j][1])

        # Preparando array com la-1 grafos (1 por layer) / Preparing an array with la-1 graphs (1 per layer).
        gr[i] = gra2

    n0 = ne[la - 1]
    aux0 = np.sum(ne[0:la - 1])
    gra3 = nx.DiGraph()

    for i3 in graf[aux0:aux0 + n0:1, 1]:

        a = find1(i3, graf[aux0:aux0 + n0:1, 1])
        # no original / original nodes
        if graf[aux0 + a, 0] == 1:
            gra3.add_node(i3, id=1, cam=la - 1, pos_i=a)
        # no incremental / incremental nodes
        if graf[aux0 + a, 0] == 0:
            gra3.add_node(i3, id=0, cam=la - 1, pos_i=a)

    # Grafo relativo a ultimo layer / graph form the last layer
    gr[la - 1] = gra3

    return s_cross, gr


"""
# verifica se posicao pode ser preenchida com no / Check if the position can be filled with 'no'

def verif_pos(grafos4, k4, layer4, no4, cont4, aux4, i4):

    # no incremental / incremental node
    if grafos4[layer4].nodes[aux4[i4 - 1]]['id'] == 0:
        aux4 = atualizap(no4, aux4, i4)
        cont4 += 1
    # no original / original node
    else:
        # Deve satisfazer restrições / Must satisfy constraints.
        if (i4 >= max(0, grafos4[layer4].nodes[aux4[i4 - 1]]['pos_i'] - k4)) and \
                (i4 <= min(grafos4[layer4].nodes[aux4[i4 - 1]]['pos_i'] + k4, aux4.shape[0] - 1)):
            aux4 = atualizap(no4, aux4, i4)
            cont4 += 1

    return cont4, aux4
"""


# idem a funcao verif_pos / otimizado
# verifica se posicao pode ser preenchida com no / Check if the position can be filled with 'no'
def check_position(grafos4, k4, layer4, no4, cont4, aux4, i4):
    node_info = grafos4[layer4].nodes[aux4[i4 - 1]]
    id_value = node_info['id']

    # no incremental / incremental node
    if id_value == 0:
        aux4 = atualizap(no4, aux4, i4)
        cont4 += 1
    # no original / original node
    elif id_value == 1:
        pos_i = node_info['pos_i']
        lower_bound = max(0, pos_i - k4)
        upper_bound = min(pos_i + k4, aux4.shape[0] - 1)

        if lower_bound <= i4 <= upper_bound:
            aux4 = atualizap(no4, aux4, i4)
            cont4 += 1

    return cont4, aux4


# Optimized / Otimizado
def calculate_crossings(u05, v05, p5, grafos5, layer5, aux5):
    control_s5 = 0

    aux5_set = set(aux5)
    p5_set = set(p5)

    for u, v in grafos5[layer5].edges():
        if u in aux5_set and v in p5_set:
            ul = find1(u, aux5)
            vl = find1(v, p5)
            control_s5 += cross(u05, ul, v05, vl)

    return control_s5


# Retorna min cruzamento e posição para inserir o nó de forma que esse mínimo ocorra
# Returns minimum crossing and position for inserting the node in a way that this minimum occurs.
def q_vq(grafos0, per0, k0, nl0, nel0, layer0, no):
    n = nel0[layer0]
    aux = np.sum(nel0[0:layer0]) if layer0 > 0 else 0

    if -1 in per0[aux:aux + n]:

        aux1 = np.copy(per0[aux:aux + n])
        ind = aux1.shape[0] - col.Counter(per0[aux:aux + n])[-1]  # list(aux1).index(-1)
        aux1[ind] = no

        # Inicializa numero de cruzamento minimo e sua posicao / Initialize minimum crossing number and its position
        min_cross = -1
        min_pos = -1

        # Primeiro Layer / First Layer
        if layer0 == 0:

            # verifica se nó possui vizinhos (sucessores) - considerar somente arestas presentes no grafo
            # Check if node has neighbors (successors) - consider only edges in the graph.
            dep = list(grafos0[layer0].neighbors(no))
            tem_e = any(ele in dep for ele in per0[aux + n:aux + n + nel0[layer0 + 1]])

            # caso o no não possua vizinhos
            # If the node does not have neighbors
            if (not dep) or (dep and not tem_e):
                min_cross = 0
                min_pos = ind

            else:

                # neigh - guarda vizinho ja inserido em p / neigh  - stores the already inserted neighbor in p
                temp_p = per0[aux + n:aux + n + nel0[layer0 + 1]]
                neigh = np.array([dep[i] for i in np.arange(len(dep)) if (dep[i] in temp_p)])

                # contabiliza os cruzamentos / compute arcs crossings
                control_s0 = 0
                u0_ = find1(no, aux1)
                for viz in neigh:
                    v0_ = find1(viz, temp_p)
                    control_s0 += calculate_crossings(u0_, v0_, temp_p, grafos0, layer0, aux1)

                # Initialize min_cross and min_pos / inicializa min_cross e min_pos
                min_cross = control_s0
                min_pos = u0_

                # Percorrem-se posições anteriores a ind para verificar se o movimento/permutação é possível
                # ind previous positions are traversed to verify if the movement/permutation is possible.
                for i in np.arange(ind, 0, -1):
                    # Controla a possibilidade de permutação / Controls the possibility of permutation
                    cont0 = 0
                    # Controla número de cruzamentos / Controls the crossings number
                    control_s0 = 0
                    # verifica se posicao pode ser preenchida com no / Check if the position can be filled with 'no'
                    (cont0, aux1) = check_position(grafos0, k0, layer0, no, cont0, aux1, i)

                    # if cont0 == 0 não mover (fica na primeira posição disponível no layer)
                    # if cont0 == 0, Don't moves (it stays in the first available position in the layer)
                    if cont0 == 0:
                        # interrompe laço for / break the for loop
                        break
                    else:
                        # contabiliza os cruzamentos / compute arcs crossings
                        u0 = find1(no, aux1)
                        for viz in neigh:
                            v0 = find1(viz, temp_p)
                            control_s0 += calculate_crossings(u0, v0, temp_p, grafos0, layer0, aux1)

                        # guarda mínimo cruzamento / Store minimum crossing
                        if control_s0 < min_cross:
                            min_cross = control_s0
                            min_pos = u0

        # Última camada / Last Layer
        if layer0 == nl0 - 1:

            ant = list(grafos0[layer0 - 1].predecessors(no))
            tem_e = any(ele in ant for ele in per0[aux - nel0[layer0 - 1]:aux])

            # caso o no nao possua predecessores / In case the node doesn't have predecessors.
            if (not ant) or (ant and not tem_e):
                min_cross = 0
                min_pos = ind

            else:
                temp_p = per0[aux - nel0[layer0 - 1]:aux]
                neigh = np.array([ant[i] for i in np.arange(len(ant)) if (ant[i] in temp_p)])

                # contabiliza os cruzamentos / compute crossings
                control_s0 = 0
                v0_ = find1(no, aux1)
                for viz in neigh:
                    u0_ = find1(viz, temp_p)
                    control_s0 += calculate_crossings(u0_, v0_, aux1, grafos0, layer0 - 1, temp_p)

                # inicializa min_cross e min_pos
                min_cross = control_s0
                min_pos = v0_

                for i in np.arange(ind, 0, -1):
                    cont0 = 0
                    control_s0 = 0
                    # verifica se posicao pode ser preenchida com no / Check if the position can be filled with 'no'
                    (cont0, aux1) = check_position(grafos0, k0, layer0, no, cont0, aux1, i)

                    # nó na primeira posição disponível na camada /it stays in the first available position in the layer
                    # interrompe laço for / break the loop for
                    if cont0 == 0:
                        break

                    else:
                        # contabiliza os cruzamentos / compute crossings
                        v0 = find1(no, aux1)
                        for viz in neigh:
                            u0 = find1(viz, temp_p)
                            control_s0 += calculate_crossings(u0, v0, aux1, grafos0, layer0 - 1, temp_p)

                        # guarda minimo cruzamento / store minimum crossing
                        if control_s0 < min_cross:
                            min_cross = control_s0
                            min_pos = v0

        # Camada central / Middle Layer
        if layer0 != 0 and layer0 != nl0 - 1:

            depo = list(grafos0[layer0].neighbors(no))
            tem_e_d = any(ele in depo for ele in per0[aux + n:aux + n + nel0[layer0 + 1]])

            antes = list(grafos0[layer0 - 1].predecessors(no))
            tem_e_a = any(ele in antes for ele in per0[aux - nel0[layer0 - 1]:aux])

            # nao possuem vizinhos anteriores ou posteriores / They don't have previous or subsequent neighbors.
            if depo == [] and antes == []:
                min_cross = 0
                min_pos = ind

            # nao possuem vizinhos inseridos em layer anterior ou posterior
            # They don't have neighbors inserted in previous or subsequent layers.
            elif (tem_e_d is False) and (tem_e_a is False):
                min_cross = 0
                min_pos = ind

            else:

                # Só possui vizinhos na camada seguinte / It only has neighbors in the next layer.
                if (tem_e_a is False) and (tem_e_d is True):

                    control_s0 = 0
                    u0_ = find1(no, aux1)
                    temp_p = per0[aux + n:aux + n + nel0[layer0 + 1]]
                    neigh = np.array([depo[i] for i in np.arange(len(depo)) if (depo[i] in temp_p)])

                    for viz in neigh:
                        v0_ = find1(viz, temp_p)
                        control_s0 += calculate_crossings(u0_, v0_, temp_p, grafos0, layer0, aux1)

                    min_cross = control_s0
                    min_pos = u0_

                    for i in np.arange(ind, 0, -1):
                        cont0 = 0
                        control_s0 = 0
                        # verifica se posicao pode ser preenchida com no / Check if the position can be filled with 'no'
                        (cont0, aux1) = check_position(grafos0, k0, layer0, no, cont0, aux1, i)

                        if cont0 == 0:
                            break

                        else:
                            u0 = find1(no, aux1)
                            for viz in neigh:
                                v0 = find1(viz, temp_p)
                                control_s0 += calculate_crossings(u0, v0, temp_p, grafos0, layer0, aux1)

                            # guarda minimo cruzamento / store minimum crossing
                            if control_s0 < min_cross:
                                min_cross = control_s0
                                min_pos = u0

                # só tem predecessores na camada anterior / It only has predecessors in the previous layer
                if (tem_e_d is False) and (tem_e_a is True):

                    temp_p = per0[aux - nel0[layer0 - 1]:aux]
                    neigh = np.array([antes[i] for i in np.arange(len(antes)) if (antes[i] in temp_p)])

                    control_s0 = 0
                    v0_ = find1(no, aux1)
                    for viz in neigh:
                        u0_ = find1(viz, temp_p)
                        control_s0 += calculate_crossings(u0_, v0_, aux1, grafos0, layer0 - 1, temp_p)

                    min_cross = control_s0
                    min_pos = v0_

                    for i in np.arange(ind, 0, -1):
                        cont0 = 0
                        control_s0 = 0
                        # verifica se posicao pode ser preenchida com no / Check if the position can be filled with 'no'
                        (cont0, aux1) = check_position(grafos0, k0, layer0, no, cont0, aux1, i)

                        if cont0 == 0:
                            break

                        else:
                            v0 = find1(no, aux1)
                            for viz in neigh:
                                u0 = find1(viz, temp_p)
                                control_s0 += calculate_crossings(u0, v0, aux1, grafos0, layer0 - 1, temp_p)

                            # guarda minimo cruzamento / store minimum crossing
                            if control_s0 < min_cross:
                                min_cross = control_s0
                                min_pos = v0

                # vizinhos nas camadas anterior e posterior/It has neighbors in both the previous and the next layers
                if (tem_e_a is True) and (tem_e_d is True):

                    # Depois / next
                    temp_pd = per0[aux + n:aux + n + nel0[layer0 + 1]]
                    neigh_d = np.array([depo[i] for i in np.arange(len(depo)) if (depo[i] in temp_pd)])
                    # Antes / previous
                    temp_pa = per0[aux - nel0[layer0 - 1]:aux]
                    neigh_a = np.array([antes[i] for i in np.arange(len(antes)) if (antes[i] in temp_pa)])

                    # controle de cruzamentos / control crossings
                    control_s0_p = 0
                    control_s0_a = 0

                    # contabiliza os cruzamentos / compute crossings
                    u0d = find1(no, aux1)
                    for vid in neigh_d:
                        v0d = find1(vid, temp_pd)
                        control_s0_p += calculate_crossings(u0d, v0d, temp_pd, grafos0, layer0, aux1)

                    v0a = find1(no, aux1)
                    for via in neigh_a:
                        u0a = find1(via, temp_pa)
                        control_s0_a += calculate_crossings(u0a, v0a, aux1, grafos0, layer0 - 1, temp_pa)

                    control_s0 = control_s0_p + control_s0_a

                    min_cross = control_s0
                    # pode ser v0a, u0d ou ind neste caso / It can be v0a, u0d, or ind in this case.
                    min_pos = v0a

                    for i in np.arange(ind, 0, -1):
                        cont0 = 0
                        # verifica se posicao pode ser preenchida com no / Check if the position can be filled with 'no'
                        (cont0, aux1) = check_position(grafos0, k0, layer0, no, cont0, aux1, i)

                        # if cont0 == 0 não posso mover (fica na primeira posição disponível)
                        # it stays in the first available position in the layer
                        if cont0 == 0:
                            break
                        else:
                            # Depois / After
                            control_s0_p = 0
                            u0d = find1(no, aux1)
                            for vid in neigh_d:
                                v0d = find1(vid, temp_pd)
                                control_s0_p += calculate_crossings(u0d, v0d, temp_pd, grafos0, layer0, aux1)

                            # Antes / Before
                            control_s0_a = 0
                            v0a = find1(no, aux1)
                            for via in neigh_a:
                                u0a = find1(via, temp_pa)
                                control_s0_a += calculate_crossings(u0a, v0a, aux1, grafos0, layer0 - 1, temp_pa)

                            control_s0 = control_s0_p + control_s0_a

                            # guarda minimo cruzamento / store minimum crossing
                            if control_s0 < min_cross:
                                min_cross = control_s0
                                # pode ser u0 ou v0 neste caso
                                min_pos = u0d

        return min_cross, min_pos

    else:
        # sem espaço no layer / There is no space in the layer
        return -1, -1


def pre_const_phase_c3(grafos1, nl1, nel1, per1, cl1, k1, alp1):
    # gera matrizes (layers x elementos) - linhas = camadas - colunas = nó incremental (espaços em branco =-1)
    # Generates matrices (layers x elements) - rows = layers - columns = incremental nodes (blank spaces = -1)
    # cr_min = menor cruzamento encontrado / cr_min_pos = posição para menor cr
    # cr_min = minimum found crossing / cr_min_pos = position for minimum cr

    # (1) Calculndo g(v) / Computing g(v)

    # Versão 1 vetorizada / Vectorized Version 1
    cr_min = (-1) * np.ones((nl1, cl1.shape[1]), dtype=int)
    cr_min_pos = (-1) * np.ones((nl1, cl1.shape[1]), dtype=int)

    for i in np.arange(nl1):
        for j in np.arange(cl1.shape[1]):
            pos = cl1[i, j]
            if pos != -1:
                min_c, pos_min_c = q_vq(grafos1, per1, k1, nl1, nel1, i, pos)
                # stop
                if min_c != -1:
                    cr_min[i, j] = min_c
                    cr_min_pos[i, j] = pos_min_c

    # calculando máximo e mínimo global
    # computing global max and min
    min_g = np.min(cr_min[cr_min != -1])
    max_g = np.max(cr_min[cr_min != -1])

    # (2) Calculando tau para tod0 nó novo / computing tau to all incremental node in candidate list and construct RCL
    min_threshold = min_g + alp1 * (max_g - min_g)
    rcl_ = np.where((cr_min != -1) & (cr_min <= min_threshold), cl1, -1)

    return cr_min, cr_min_pos, rcl_


# Chamada seguinte - Retorna RCL, min pos, min cross and p atualizado
# THE FOLLOW CALL - Return RCL, min pos, min cross and p updated
def pre_const_phase_c3_2(grafos3, nl3, nel3, per3, cl3, k3, alp3, layers3, cr_min3, cr_min_pos3):
    # (1) Computing g(v)
    # percorre camadas / Sweep through
    for i in layers3:
        # calcula min cross e melhor posição para todos os nós incrementais em layers3
        # compute minimum crossing and best position for all incremental nodes in layers3
        for j in np.arange(cl3.shape[1]):
            if cl3[i][j] != -1:
                min_c3, pos_min_c3 = q_vq(grafos3, per3, k3, nl3, nel3, i, cl3[i][j])
                # Sem espaço para inserção de novos nós / No space available for inserting new nodes
                if min_c3 == -1:
                    cr_min3[i][j] = -1
                    cr_min_pos3[i][j] = -1
                if min_c3 != -1:
                    cr_min3[i][j] = min_c3
                    cr_min_pos3[i][j] = pos_min_c3

    # calcula maximo e minimo global / computing global max and min
    # exclui -1 / Exclude the value -1
    masked_array = np.ma.masked_equal(cr_min3, -1)
    # calculo max e min valores pos mascara / Calculate the minimum and maximum value among the non-masked values
    min_g3 = np.min(masked_array)
    max_g3 = np.max(masked_array)

    # (2) calcula tau para tod0 no incremental na lista de candidatos e costroi RCL
    # (2) computing tau to all incremental node in candidate list and construct RCL
    """
    # versao 0
    rcl3 = np.array([[cl3[i][j] if (cr_min3[i][j] != -1 and cr_min3[i][j] <= min_g + alp3 * (max_g - min_g)) else
                      -1 for j in np.arange(cl3.shape[1])] for i in np.arange(nl3)])
    """
    # versao vetorizada / Vectorized Version
    rcl_mask = np.logical_and(cr_min3 != -1, cr_min3 <= min_g3 + alp3 * (max_g3 - min_g3))
    rcl_values = np.where(rcl_mask, cl3, -1)
    rcl3 = rcl_values

    return cr_min3, cr_min_pos3, rcl3, per3


# Construcao da solucao inicial /Construction of an initial solution
def const_phase_c3(grafos2, nl2, nel2, per2, cl2, k2, alp2, cro_min2, cro_min_pos2, rcl2):
    # Guarda número de cruzamentos inseridos após a inserção do novo nó
    # Keep the number of crossings inserted from incremental nodes inserted
    # parc_cross = 0

    while -1 in per2:

        # Seleciona elemento na RCL aleatóriamente / select node v* randomly in RCL
        """
        Este código encontra os índices onde rcl2 não é igual a -1 usando np.where(). Em seguida, verifica se existem 
        índices não vazios. Se houver, ele seleciona um índice aleatório desses índices não vazios e o usa para acessar
         a posição correspondente no array rcl2.

        This code finds the indices where rcl2 is not equal to -1 using np.where(). Then, it checks if there are any 
        non-empty indices. If there are, it selects a random index from those non-empty indices and uses it to access 
        the corresponding position in the rcl2 array.
        """
        # inicializando / initialization
        x = None
        y = None
        # procura indices nao vazios / Find non-empty indices
        non_empty_indices = np.where(rcl2 != -1)

        if non_empty_indices[0].size > 0:
            random_index = np.random.randint(non_empty_indices[0].size)
            x = non_empty_indices[0][random_index]
            y = non_empty_indices[1][random_index]

        # identifica posicao original do no / identify the original position of node cd[x][y] in Gr/p
        n = nel2[x]
        aux = np.sum(nel2[0:x]) if x > 0 else 0

        # -----
        # Usa arrays numpy / Use arrays numpy
        # Encontre o índice do primeiro -1 na lista / Find the index of the first -1 in the list.
        ind = np.where(per2[aux:aux + n] == -1)[0][0]
        # Crie uma cópia de per2 para evitar modificações indesejadas no array original
        # Create a copy of per2 to prevent undesired modifications to the original array.
        temp_p = np.copy(per2)
        # Remova a primeira ocorrência de -1 na camada da lista
        # Remove the first occurrence of -1 in the layer of the list.
        temp_p = np.delete(temp_p, aux + ind)
        # Calcule a posição onde o novo elemento será inserido
        # Calculate the position where the new element will be inserted.
        insert_position = cro_min_pos2[x][y] + aux
        # Insira o novo elemento na posição calculada e atualize p
        # Insert the new element at the calculated position and update p.
        up_p = np.insert(temp_p, insert_position, cl2[x][y])

        # -----

        # atualiza cruzamentos / update crossings
        # parc_cross += cro_min2[x][y]

        # atualiza lista de candidatos / update cd
        cl2[x][y] = -1
        # atualizacoes / updates:
        cro_min2[x][y] = -1
        cro_min_pos2[x][y] = -1

        #  recalcula g e tu, reconstroi RCL /recompute g and tau, rebuild RCL

        # primeiro layer (atualiza layer atual e proximo) / first layer (update current and next layers)
        if x == 0:
            (cro_min2, cro_min_pos2, rcl2, per2) = pre_const_phase_c3_2(grafos2, nl2, nel2, up_p, cl2, k2, alp2,
                                                                        [x, x + 1], cro_min2, cro_min_pos2)

        # ultimo layer (update layers atual e anterior) / Last Layer (update current and previous layers)
        if x == nl2 - 1:
            (cro_min2, cro_min_pos2, rcl2, per2) = pre_const_phase_c3_2(grafos2, nl2, nel2, up_p, cl2, k2, alp2,
                                                                        [x - 1, x], cro_min2, cro_min_pos2)

        # layer central (update layers atual, anterior e posterior)
        # middle layer (update current, previous and next layer)
        if (x != 0) and (x != nl2 - 1):
            (cro_min2, cro_min_pos2, rcl2, per2) = pre_const_phase_c3_2(grafos2, nl2, nel2, up_p, cl2, k2, alp2,
                                                                        [x - 1, x, x + 1], cro_min2, cro_min_pos2)

    return per2  # parc_cross


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


# u_ e v_ arrays com as posicoes das arestas do grafo / Arrays with the positions of the edges in the graph.
# u1 outro no - identificacao para evitar contabilizacoes duplicadas / Identification to avoid duplicate counts.
# (u0,v0) aresta analisada / Analyzed edge - c_jo guarda cruzamentos / Stores intersections
def comp_1(u_, v_, u0, v0, u1, c_j):
    for kk1 in np.arange(np.size(u_)):
        # para nao contabilizar esse(s) cruzamento(s) duas vezes
        # To avoid counting this/these intersection(s) twice
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
        # para nao contabilizar esse(s) cruzamento(s) duas vezes
        # To avoid counting this/these intersection(s) twice
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


# inicializa variaveis / variable initialization
def begin_f(graph_, nel_, nl_):
    # Construção da lista de candidatos / array com nós incrementais - linhas representam layers
    # contruction of candidate list / Array with incremental nodes - rows represent layers
    cl_ = np.array(cand_list(graph_[0:, 0:2:1], nl_, nel_))
    # print('cl=', cl)

    # Retorna o cruzamento inicial e nl-1 digrafos (networkx)
    # Returns the initial crossing and nl-1 digraphs (NetworkX)
    ini_cross_, grafos_ = fc(graph_, nl_, nel_, cl_)
    # print('ini Cross=', ini_cross_)
    # print(grafos_[1].nodes[4]['pos_i'])
    # print('BRUNA - id=', gra.nodes[5]['id'], 'cam=', gra.nodes[5]['cam'], 'pos_i=', gra.nodes[5]['pos_i'])

    # Vetor de permutação / permutation vector
    per_ = np.array(perm2(graph_[0:, 0:2:1]))
    # print('per ini=', per_)
    # print(' ')

    return cl_, ini_cross_, grafos_, per_


# fase construtiva grasp
def greedy_rand_construction(grafos_t, nlt, nelt, pert, clt, kt, alphat):
    # Primeira chamada a funcao pre_const_phase_c3 / The first call to pre_const_phase_c3  function
    (cro_mf, cro_min_pf, rcl_f) = pre_const_phase_c3(grafos_t, nlt, nelt, pert, clt, kt, alphat)

    # chamad a funcao const_phase_c3 / The call to const_phase_c3 function
    (per_const) = const_phase_c3(grafos_t, nlt, nelt, pert, clt, kt, alphat, cro_mf, cro_min_pf, rcl_f)
    # print('p=', per_cons[0])

    # contabilizacao de cruzamento parcial usando p - sem cruzamentos duplicados
    # Partial intersection tally using p - without duplicate intersections.
    cro_pt = cro_parc_final(grafos_t, nlt, nelt, per_const)
    # print('Cruzamentos inseridos =', cro_p)
    # print('p=', per_cons)

    return per_const, cro_pt


# melhorias grasp / improvement grasp
def local_search_f(grafosw, nlw, nelw, per_consw, cro_pw, ini_crossw, kw):
    # busca local 1 / local search part 1 - swap function - devolve/return total cros pos swap
    per_new, cro_tot_new = swap_func(grafosw, nlw, nelw, per_consw, cro_pw, ini_crossw)
    # print('Total cross new=', cro_tot_new)
    # print('P new=', per_new)

    # busca local 2 / local search part 2 - insertion function
    # contabilizacao/compute cros pos swap
    cro_p2 = cro_parc_final(grafosw, nlw, nelw, per_new)
    # Resultado final/ Final result  per_f e cro_f
    per_fw, c_fw = insertion_func(grafosw, nlw, nelw, cro_p2 + ini_crossw, per_new, kw, ini_crossw)
    cro_fw = cro_parc_final(grafosw, nlw, nelw, per_fw) + ini_crossw
    # print(ini_cross + cro_p, cro_f + ini_cross)
    # print('compara=', c_f, cro_f + ini_cross)

    return per_fw, cro_fw


# grasp
def grasp(grafos_v, nlv, nel_v, perv, clv, kv, alphav, ini_cro_v):
    # fase construtiva / Construction phase.
    per_cons_v, cro_pv = greedy_rand_construction(grafos_v, nlv, nel_v, perv, clv, kv, alphav)

    # busca local / Local Search
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
        minx = cont(graf_x[aux:aux + n])
        if minx < min_x:
            min_x = minx

    return min_x


# MAIN()------------------------------------------------------------------------------##

def main():
    # Leitura dos dados / Reading the data
    all_graphs = data_read()

    # Total de execuções - define / total runs 30 - DEFINE
    total_run = int(input("Runs number:"))
    # Total de iterações GRASP - define / GRASP Iteration 100 - DEFINE
    it_ma = int(input("GRASP Iteration:"))
    # stop critter 20 - DEFINE
    stop_ = int(input("Stop Critter:"))

    for ru in np.arange(total_run):

        # keep the crossings numbers constructive and final phase
        arq_c3 = open('res_c3/C3_' + str(ru+1) + '.txt', 'a')
        # keep the permutation value
        arq_c3_p = open('res_c3/C3_p_' + str(ru+1) + '.txt', 'a')
        # keep the permutation value pos constructive phase
        arq_c3_con_p = open('res_c3/C3_const_p_' + str(ru+1) + '.txt', 'a')
        # keep grasp iterations
        arq_c3_iter = open('res_c3/C3_iter_' + str(ru+1) + '.txt', 'a')

        # the number of incremental nodes per layer must be greater than or equal to k.
        k_all = [1, 2, 3]
        for k in k_all:

            # percorre instâncias (arrays) da lista all_graphs /Iterating through the instances of the all_graphs list
            for ind_, (graph, nel, nl) in enumerate(all_graphs):
                # print("Graph:", graph, "NEL:", nel, "NL:", nl)
                # print("----------------------")

                # controla núm nós incr >= k / control node number >= k
                t_k = test_k(graph, nel, nl)
                if t_k < k:
                    arq_c3.write(str(ind_) + str(',') + str(k) + str(',') + str(-1) + str(',') + str(-1) + str(',')
                                 + str(-1) + '\n')
                    arq_c3_p.write(str(ind_) + str(',') + str(k) + str(',') + str(-1) + '\n')
                    arq_c3_con_p.write(str(ind_) + str(',') + str(k) + str(',') + str(-1) + '\n')

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

                        # inicializa variaveis / variable initialization
                        cl, ini_cross, grafos, per = begin_f(graph, nel, nl)
                        # chamada grasp
                        x_, f_x, p_con, cro_con = grasp(grafos, nl, nel, per, cl, k, alpha, ini_cross)

                        stop_aux += 1

                        # seleciona melhor solucao / select the best solution
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
                        arq_c3_iter.write(str(ind_) + str(',') + str(k) + str(',') + str(round(alpha, 2)) + str(',') +
                                          str(ite + 1) + str(',') + str(best_cost_con) + str(',') + str(best_cost)
                                          + str(',') + str(timedelta(seconds=end_time_in - start_time_in)) + '\n')

                    end_time = time.monotonic()

                    # id graph, k, f(x) constructive phase, f(x) final, time
                    arq_c3.write(
                        str(ind_) + str(',') + str(k) + str(',') + str(best_cost_con) + str(',') + str(best_cost)
                        + str(',') + str(timedelta(seconds=end_time - start_time)) + '\n')
                    arq_c3_p.write(str(ind_) + str(',') + str(k) + str(',') + str(best_solution) + '\n')
                    arq_c3_con_p.write(str(ind_) + str(',') + str(k) + str(',') + str(best_solution_con) + '\n')

                # break

        arq_c3.close()
        arq_c3_p.close()
        arq_c3_con_p.close()
        arq_c3_iter.close()


# Executa o programa
if __name__ == "__main__":
    main()
